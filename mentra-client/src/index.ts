import { AppServer, AppSession, AuthenticatedRequest} from "@mentra/sdk";
import express from 'express';
import path from 'path'
import {spawn, ChildProcess} from 'child_process'
import fs from 'fs'
import cors from 'cors'
// import { error, timeStamp } from "console";
// import { connect } from "http2";


// env conf
const PACKAGE_NAME = process.env.PACKAGE_NAME ?? (() => {throw new Error('PACKAGE_NAME is not set in the .env'); })(); // the () at the end are for imediate execution and to not make PACKAGE_NAME a function hah | that s called ImediInvkedFuncExpression IIFE
const MENTRA_OS_API_KEY = process.env.MENTRA_OS_API_KEY ?? (() => {throw new Error('MENTRA_OS_API_KEY is not set in the .env');})();
const PORT = parseInt(process.env.PORT || '3000')
const PYTHON_SCRIPT_PATH = process.env.PYTHON_SCRIPT_PATH ?? (() => {throw Error('PYTHON_SCRIPT_PATH is not set in the .env');})();
const REACT_BUILD_PATH = process.env.REACT_BUILD_PATH ?? (() => {throw Error('REACT_BUILD_PATH is not set in the .env');})();
const PROCESSING_TIMEOUT = parseInt(process.env.PROCESSING_TIMEOUT || '300000')

interface ProcessingResult {
    sessionId: string;
    timestamp: number;
    results: any;
    status: 'processing' | 'completed' | 'error';
    message?: string;
}

interface ProcessingSession {
    sessionId: string;
    userId: string;
    pythonProcess?: ChildProcess;
    startTime: number;
    streamUrl?: string;
}


class SeenittApp extends AppServer {
    // Active SSE connections by userId (in the case where the user opens multiple tabs etc ...)
    private sseConnections = new Map<string, express.Response[]>();

    // Active processing sessions
    private processingSessions = new Map<string, ProcessingSession>();

    // Results cache
    private resultsCache = new Map<string, ProcessingResult[]>();


    constructor(){
        super({
            packageName: PACKAGE_NAME,
            apiKey: MENTRA_OS_API_KEY, 
            port: PORT,
        });

        this.setupRoutes();
        // this.validatePaths();
    }


    // setups Express routes and middleware 
    private setupRoutes(): void {
        const app = this.getExpressApp();

        // CORS for dev
        if (process.env.NODE_ENV != 'production') {
            app.use(cors({
                origin: ['http://localhost:5173', 'http://localhost:3000'],
                credentials: true
            }));
        }

        app.use(express.json());


        // Serve React App (production)
        if (process.env.NODE_ENV === 'production' && fs.existsSync(path.resolve(REACT_BUILD_PATH))){
            app.use(express.static(path.resolve(REACT_BUILD_PATH))); // Turn the Express server into a file server for React files
        }

        // Health check - a GET endpoint 
        app.get('/api/health', (req, res) => {
            res.json({
                status: 'ok',
                timestamp: Date.now(),
                processingSessions: this.processingSessions.size
            });
        });

        // SSE endpoint for realtime updates with the restapp
        app.get('/api/stream', (req: AuthenticatedRequest, res) => {
            const userId = req.authUserId;

            if (!userId){
                res.status(401).json({error: 'Unauthorized'});
                return;
            }


            // SSE headers
            res.writeHead(200, {
                'Content-type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
            });

            // Store the connection
            if (!this.sseConnections.has(userId)) {
                this.sseConnections.set(userId, []);
            }
            this.sseConnections.get(userId)!.push(res);

            // Send connection confirmation
            res.write(`data: ${JSON.stringify({
                type: 'connected',
                userId, 
                timestamp: Date.now()    
            })}\n\n`);



            // cleanup on disconnect
            req.on('close', () => {this.removeSSEConnection(userId, res)});


        });



        // res.json()	Auto stringify + headers + send
        // JSON.stringify()	Just converts to string

        // Get processing results
        app.get('/api/results/:sessionId', (req: AuthenticatedRequest, res) => {
            const { sessionId } = req.params; // :sessionId -> put whatever comes here into req.params, then this is the moder way to destructure istead of req.params.sessionId
            const results = this.resultsCache.get(sessionId) || [];
            res.json({sessionId, results});
        });



        // Manual processing -- nah 
        app.post('/api/process', (req: AuthenticatedRequest, res) => {
            const userId = req.authUserId;
            if (!userId){
                res.status(401).json({error: 'Unauthorized'})
                return;
            }

            const { streamUrl, sessionId } = req.body;
            if(streamUrl && sessionId) {
                this.startProcessing(sessionId, userId, streamUrl);
                res.json({status: 'started', sessionId});
            }
            else {
                res.status(400).json({error: "streamUrl and sessionId required"})
            }
        });


        // react app for all other routes #SinglePageApplication
        if (process.env.NODE_ENV === 'production') {
            app.get('*', (req, res) => {
                const index_path = path.resolve(REACT_BUILD_PATH, 'index.html');
                if (fs.existsSync(index_path)){
                    res.sendFile(index_path);
                }
                else {
                    res.status(404).json({error: 'React app not built or index.html path error ...'})
                }
            });
        } 
    }


    // time to handle new MentraOS session
    protected async onSession(session: AppSession, sessionId: string, userId: string): Promise<void> {
        session.logger.info(`New session: ${sessionId} for user: ${userId}`);
    
        // store session info
        this.processingSessions.set(sessionId, {
            sessionId,
            userId,
            startTime: Date.now()
        });

        // now we start the camera stream in a managed way for the moment
        const stream = await session.camera.startManagedStream();
        session.logger.info(`Camera stream started: ${stream.hlsUrl}`);

        // update session with streaminfo
        const sessionInfo = this.processingSessions.get(sessionId);
        sessionInfo!.streamUrl = stream.hlsUrl;

        // start the python processing pipeline
        this.startProcessing(sessionId, userId, stream.hlsUrl);


        // send initial status to web client
        this.broadcastToUser(userId, {
            type: 'session_started',
            sessionId,
            streamUrl: stream.hlsUrl,
            timeStamp: Date.now()
        });



        session.events.onDisconnected(() => {
            session.logger.info(`Session ${sessionId} disconnected`);
            this.cleanupSession(sessionId);
        });

    }


    // func to start the python processing pipeline
    private startProcessing(sessionId: string, userId: string, streamUrl: string): void {
        const sessionInfo = this.processingSessions.get(sessionId);
        if (!sessionInfo) return;
    
        console.log(`Starting python processing for session ${sessionId}`);

        // spawn python process
        const pythonProcess = spawn('python3', [
            path.resolve(PYTHON_SCRIPT_PATH),
            '--stream-url', streamUrl,
            '--session-id', sessionId,
            '--output-format', 'json'
        ]);

        sessionInfo.pythonProcess = pythonProcess;
    
    
        // getting the output
        pythonProcess.stdout.on('data', (data) => {
            try {
                const lines = data.toString().trim().split('\n'); // toSing becasue it comes as buffer binary, trim to remove whitesapaces + \n in the edges
                lines.forEach((line: string) => {
                    if (line.trim()){
                        const result = JSON.parse(line);
                        this.handleProcessingResult(sessionId, userId, result);
                    }
                });
            } catch(error) {
                console.error('Error parsing python output', error);
            }
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python error: ${data}`);
            this.broadcastToUser(userId, {
                type: 'processing_error',
                message: data.toString(),
                timestamp: Date.now()
            });
        });

        pythonProcess.on('close', (code) => {
            console.error(`Python process exited with code: ${code}`);
            if (sessionInfo.pythonProcess === pythonProcess) {
                sessionInfo.pythonProcess = undefined;
            }
        });



        setTimeout(() => {
            if (pythonProcess && !pythonProcess.killed) {
                pythonProcess.kill();
                console.log(`Timeout - Python process killed for ${sessionId}`);
            }
        }, PROCESSING_TIMEOUT);
    
    }


    private handleProcessingResult(sessionId: string, userId: string, result: any): void {
        const processedResult: ProcessingResult = {
            sessionId,
            timestamp: Date.now(),
            results: result,
            status: result.status || 'completed'
        };
        
        
        // cache results
        if (!this.resultsCache.has(sessionId)) {
            this.resultsCache.set(sessionId, []);
        }
        this.resultsCache.get(sessionId)!.push(processedResult);

        // keep it s size manageable
        const cache = this.resultsCache.get(sessionId)!;
        if (cache.length > 100) {
            cache.splice(0, cache.length - 100);
        }

        // broadcast to webapp clients
        this.broadcastToUser(userId, {
            type: 'processing_result',
            ...processedResult // the ... is Spread
        });
        
        console.log(`Processing result for ${sessionId}:`, result);
        
    }


    // the broadcastToUser def
    private broadcastToUser(userId: string, data: any){
        const connections = this.sseConnections.get(userId);
        if (connections) {
            const message = `data: ${JSON.stringify(data)}\n\n`;

            connections.forEach(res => {
                try {
                    res.write(message);
                } catch(error) {
                    console.error('Error writing to SSE connection:', error);
                }
            });
        }
    }


    private removeSSEConnection(userId: string, res: express.Response): void {
        const connections = this.sseConnections.get(userId);
        if (connections) {
            const index = connections.indexOf(res);
            if (index > -1) {
                connections.splice(index, 1);
            }
            if (connections.length === 0) {
                this.sseConnections.delete(userId);
            }
        }
    }

    private cleanupSession(sessionId: string): void {
        const sessionInfo = this.processingSessions.get(sessionId);
        if (sessionInfo) {
            // Kill python process before to go
            if (sessionInfo.pythonProcess && !sessionInfo.pythonProcess.killed) {
                sessionInfo.pythonProcess.kill();
            }       

            // Kill SSE connections 
            const connections = this.sseConnections.get(sessionInfo.userId);
            if (connections) {
                connections.forEach(res => res.end());
            }

            // remove from active sessions
            this.processingSessions.delete(sessionId);

            console.log(`Cleaned up session ${sessionId}`)  
        }
    }
}



// Satrt the server
const app = new SeenittApp();
app.start().catch(console.error);


// garceful shutdown
process.on('SIGINT', () => {
    console.log('Shutting down ...')
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('Shutting down ...')
    process.exit(0);
});


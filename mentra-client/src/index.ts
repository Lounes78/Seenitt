import { AppServer, AppSession, AuthenticatedRequest} from "@mentra/sdk";
import express from 'express';
import path from 'path'
import {spawn, ChildProcess} from 'child_process'
import fs from 'fs'
import cors from 'cors'


// env conf
const PACKAGE_NAME = process.env.PACKAGE_NAME ?? (() => {throw new Error('PACKAGE_NAME is not set in the .env'); })(); // the () at the end are for imediate execution and to not make PACKAGE_NAME a function hah | that s called ImediInvkedFuncExpression IIFE
const MENTRA_OS_API_KEY = process.env.MENTRA_OS_API_KEY ?? (() => {throw new Error('MENTRA_OS_API_KEY is not set in the .env');})();
const PORT = parseInt(process.env.PORT || '3000')
const PYTHON_SCRIPT_PATH = process.env.PYTHON_SCRIPT_PATH ?? (() => {throw Error('PYTHON_SCRIPT_PATH is not set in the .env');})();
const REACT_BUILD_PATH = process.env.REACT_BUILD_PATH ?? (() => {throw Error('REACT_BUILD_PATH is not set in the .env');})();
const PROCESSING_TIMEOUT = (process.env.PROCESSING_TIMEOUT || '300000')

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
    private activeSessions = new Map<string, ProcessingSession>();

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
        const app = this.getExpressApp;

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
                activeSessions: this.activeSessions.size
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
    








}



package com.seenitt

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.meta.wearable.dat.camera.StreamSession
import com.meta.wearable.dat.camera.startStreamSession
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.StreamSessionState
import com.meta.wearable.dat.camera.types.VideoFrame
import com.meta.wearable.dat.camera.types.VideoQuality
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.AutoDeviceSelector
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import com.meta.wearable.dat.core.types.RegistrationState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

// CONFIGURATION CONSTANTS
private const val TARGET_FPS = 5
private const val SERVER_URL = "wss://wildly-knowing-adder.ngrok-free.app"
private const val TAG = "DEBUG_STREAM"

class MainActivity : ComponentActivity() {

    private lateinit var btnConnect: Button
    private lateinit var btnStream: Button
    private lateinit var btnTogglePreview: Button
    private lateinit var tvStatus: TextView
    private lateinit var ivPreview: ImageView

    private var streamSession: StreamSession? = null
    
    // Jobs
    private var videoJob: Job? = null
    private var stateJob: Job? = null 
    private var timeoutJob: Job? = null

    private val sessionMutex = Mutex()
    private var h264Encoder: AvcEncoder? = null
    private var isPreviewEnabled = true
    
    @Volatile
    private var isCommandRunning = false

    // --- PERMISSION LAUNCHERS ---
    private val permissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) {
            Log.d(TAG, "Android Permissions Granted")
            initializeWearables()
        } else {
            Log.e(TAG, "Android Permissions Denied")
            updateStatus("Permissions Missing", Color.RED)
        }
    }

    private val wearablesPermissionLauncher = registerForActivityResult(
        Wearables.RequestPermissionContract()
    ) { result ->
        if (result == PermissionStatus.Granted) {
            Log.d(TAG, "Glasses Camera Permission Granted via Prompt")
            triggerStartSequence()
        } else {
            Log.e(TAG, "Glasses Camera Permission Denied")
            updateStatus("Camera Denied on Glasses", Color.RED)
            resetUI(canStream = true)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnConnect = findViewById(R.id.btn_connect)
        btnStream = findViewById(R.id.btn_stream)
        btnTogglePreview = findViewById(R.id.btn_toggle_preview)
        tvStatus = findViewById(R.id.tv_status)
        ivPreview = findViewById(R.id.iv_preview)

        btnConnect.setOnClickListener {
            Log.d(TAG, "Click: Connect/Register")
            updateStatus("Registering...", Color.YELLOW)
            Wearables.startRegistration(this)
        }

        btnStream.setOnClickListener {
            if (isCommandRunning) return@setOnClickListener
            
            lifecycleScope.launch {
                if (streamSession != null) {
                    performStopStreaming()
                } else {
                    checkPermissionsAndStart()
                }
            }
        }

        btnTogglePreview.setOnClickListener {
            isPreviewEnabled = !isPreviewEnabled
            btnTogglePreview.text = if (isPreviewEnabled) "Preview: ON" else "Preview: OFF"
            if (!isPreviewEnabled) ivPreview.setImageDrawable(null)
        }

        btnStream.isEnabled = false
        checkAndroidPermissions()
    }

    private fun updateStatus(text: String, color: Int = Color.BLACK) {
        runOnUiThread {
            tvStatus.text = text
            tvStatus.setTextColor(color)
        }
    }

    private fun resetUI(canStream: Boolean) {
        isCommandRunning = false
        runOnUiThread {
            btnStream.isEnabled = canStream
            btnStream.text = if (streamSession != null) "Stop Streaming" else "Start Streaming"
            if (streamSession == null) {
                updateStatus("Ready", Color.parseColor("#006400"))
            }
        }
    }

    private fun checkAndroidPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.INTERNET,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.NEARBY_WIFI_DEVICES)
        }

        if (permissions.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            initializeWearables()
        } else {
            permissionsLauncher.launch(permissions.toTypedArray())
        }
    }

    private fun initializeWearables() {
        Wearables.initialize(this)
        lifecycleScope.launch {
            Wearables.registrationState.collect { state ->
                Log.d(TAG, "Registration State: $state")
                when (state) {
                    is RegistrationState.Registered -> {
                        updateStatus("Device Paired & Ready", Color.parseColor("#006400"))
                        btnConnect.isEnabled = false
                        btnConnect.text = "Connected"
                        btnStream.isEnabled = true
                    }
                    else -> {
                        updateStatus("Status: ${state::class.simpleName}", Color.GRAY)
                        btnConnect.isEnabled = true
                        btnStream.isEnabled = false
                    }
                }
            }
        }
    }

    private fun checkPermissionsAndStart() {
        isCommandRunning = true
        btnStream.isEnabled = false
        btnStream.text = "Checking..."

        lifecycleScope.launch {
            try {
                Log.d(TAG, "Checking camera permission...")
                val status = Wearables.checkPermissionStatus(Permission.CAMERA)
                Log.d(TAG, "Camera permission status: $status")
                
                if (status == PermissionStatus.Granted) {
                    triggerStartSequence()
                } else {
                    Log.w(TAG, "Camera permission not granted, requesting...")
                    wearablesPermissionLauncher.launch(Permission.CAMERA)
                }
            } catch (e: Exception) {
                Log.e(TAG, "CRITICAL ERROR in Permission Check", e)
                updateStatus("Permission Check Failed: ${e.message}", Color.RED)
                resetUI(canStream = true)
            }
        }
    }

    private fun triggerStartSequence() {
        lifecycleScope.launch {
            updateStatus("Initializing...", Color.BLUE)
            delay(500) 
            performStartStreaming()
        }
    }
    
    private suspend fun performStartStreaming() {
        sessionMutex.withLock {
            try {
                if (streamSession != null) {
                    Log.w(TAG, "Stream session already exists, aborting")
                    return@withLock
                }

                Log.d(TAG, "=== START STREAMING SEQUENCE ===")
                
                // 1. Connect Network
                Log.d(TAG, "Step 1: Connecting to WebSocket...")
                withContext(Dispatchers.IO) {
                    ServerClient.connect(SERVER_URL)
                }
                Log.d(TAG, "Step 1: WebSocket connection initiated")

                // 2. Start Session
                Log.d(TAG, "Step 2: Starting stream session...")
                val session = Wearables.startStreamSession(
                    this@MainActivity,
                    AutoDeviceSelector(),
                    StreamConfiguration(VideoQuality.MEDIUM, TARGET_FPS)
                )
                streamSession = session
                Log.d(TAG, "Step 2: Stream session created: $session")
                updateStatus("LIVE STREAMING @ 5 FPS", Color.RED)

                var hasStarted = false
                var streamingDetected = false

                // 3. Timeout mechanism
                timeoutJob = lifecycleScope.launch {
                    delay(15000) // 15 second timeout
                    if (!streamingDetected) {
                        Log.e(TAG, "⚠️ TIMEOUT: Stream never started after 15 seconds!")
                        updateStatus("Stream Timeout - Check Glasses", Color.RED)
                        performStopStreaming()
                    }
                }

                // 4. Monitor State Changes
                stateJob = lifecycleScope.launch {
                    Log.d(TAG, "Step 3: State monitoring job started")
                    try {
                        session.state.collect { state ->
                            Log.d(TAG, ">>> STATE CHANGE: $state <<<")
                            
                            if (state == StreamSessionState.STARTING) {
                                Log.d(TAG, "Stream is STARTING...")
                                updateStatus("Starting Stream...", Color.YELLOW)
                            }
                            
                            if (state == StreamSessionState.STREAMING) {
                                hasStarted = true
                                streamingDetected = true
                                timeoutJob?.cancel()
                                Log.d(TAG, "✓ Stream is now STREAMING (hasStarted = true)")
                                updateStatus("LIVE STREAMING @ 5 FPS", Color.RED)
                            }
                            
                            if (state == StreamSessionState.STOPPED) {
                                Log.w(TAG, "Stream STOPPED (hasStarted was: $hasStarted)")
                                if (hasStarted) {
                                    Log.w(TAG, "Stream stopped unexpectedly!")
                                    updateStatus("Stream Stopped Unexpectedly", Color.RED)
                                    launch { performStopStreaming() }
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in state collection", e)
                    }
                    Log.d(TAG, "State monitoring job ended")
                }

                // 5. Collect Video Frames
                videoJob = lifecycleScope.launch(Dispatchers.Default) {
                    Log.d(TAG, "Step 4: Video collection job started")
                    var frameCount = 0
                    try {
                        session.videoStream.collect { frame ->
                            frameCount++
                            if (frameCount == 1) {
                                Log.d(TAG, "!!! FIRST FRAME RECEIVED !!! Size: ${frame.width}x${frame.height}")
                                updateStatus("STREAMING - Frames Received", Color.GREEN)
                            }
                            if (frameCount % 25 == 0) {
                                Log.d(TAG, "Video frames received: $frameCount")
                            }
                            processFrame(frame)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error in video stream collection", e)
                    }
                    Log.w(TAG, "Video stream collection ended. Total frames: $frameCount")
                }

                Log.d(TAG, "=== STREAMING SETUP COMPLETE ===")

            } catch (e: Exception) {
                Log.e(TAG, "START FAILURE", e)
                updateStatus("Error: ${e.message}", Color.RED)
                forceCleanup()
            } finally {
                isCommandRunning = false
                runOnUiThread {
                    btnStream.isEnabled = true
                    btnStream.text = if (streamSession != null) "Stop Streaming" else "Start Streaming"
                }
            }
        }
    }

    private suspend fun performStopStreaming() {
        sessionMutex.withLock {
            Log.d(TAG, "=== STOP STREAMING ===")
            forceCleanup()
            resetUI(canStream = true)
        }
    }

    private fun forceCleanup() {
        Log.d(TAG, "Force cleanup started")
        
        timeoutJob?.cancel()
        timeoutJob = null
        
        stateJob?.cancel()
        stateJob = null
        
        videoJob?.cancel()
        videoJob = null

        try {
            streamSession?.close()
            Log.d(TAG, "Stream session closed")
        } catch (e: Exception) { 
            Log.w(TAG, "Cleanup Session Error", e) 
        }
        streamSession = null

        try {
            h264Encoder?.close()
            Log.d(TAG, "Encoder closed")
        } catch (e: Exception) { 
            Log.w(TAG, "Cleanup Encoder Error", e) 
        }
        h264Encoder = null

        ServerClient.close()

        runOnUiThread {
            ivPreview.setImageDrawable(null)
            updateStatus("Ready", Color.parseColor("#006400"))
        }
        
        Log.d(TAG, "Force cleanup completed")
    }

    private fun processFrame(frame: VideoFrame) {
        val rawData = extractRawI420(frame) ?: run {
            Log.w(TAG, "extractRawI420 returned null for frame ${frame.width}x${frame.height}")
            return
        }
        
        val w = frame.width
        val h = frame.height

        // 1. Check if Encoder exists AND matches current resolution
        if (h264Encoder != null && !h264Encoder!!.isSize(w, h)) {
            Log.w(TAG, "Resolution changed from ${h264Encoder!!.width}x${h264Encoder!!.height} to ${w}x${h}. Restarting Encoder.")
            h264Encoder?.close()
            h264Encoder = null
        }

        // 2. Initialize Encoder if null
        if (h264Encoder == null) {
            h264Encoder = AvcEncoder(w, h) { h264Bytes ->
                ServerClient.sendFrame(h264Bytes)
            }
        }
        
        // 3. Encode (Non-blocking)
        try {
            h264Encoder?.encode(rawData)
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error", e)
        }

        // 4. Update Preview
        if (isPreviewEnabled) {
            updateLocalPreview(rawData, w, h)
        }
    }

    // --- UTILS ---
    private fun extractRawI420(videoFrame: VideoFrame): ByteArray? {
        val buffer = videoFrame.buffer
        val width = videoFrame.width
        val height = videoFrame.height
        val expectedSize = (width * height * 1.5).toInt()
        
        if (buffer.remaining() < expectedSize) {
            Log.w(TAG, "Buffer size mismatch: expected $expectedSize, got ${buffer.remaining()}")
            return null
        }

        val byteArray = ByteArray(buffer.remaining())
        val originalPosition = buffer.position()
        buffer.get(byteArray)
        buffer.position(originalPosition)
        return byteArray
    }

    private fun updateLocalPreview(i420: ByteArray, width: Int, height: Int) {
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val nv21 = convertI420toNV21(i420, width, height)
                val image = YuvImage(nv21, ImageFormat.NV21, width, height, null)
                val out = ByteArrayOutputStream()
                image.compressToJpeg(Rect(0, 0, width, height), 50, out)
                val jpegBytes = out.toByteArray()
                val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
                
                withContext(Dispatchers.Main) {
                    ivPreview.setImageBitmap(bitmap)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Preview update error", e)
            }
        }
    }

    private fun convertI420toNV21(input: ByteArray, width: Int, height: Int): ByteArray {
        val output = ByteArray(input.size)
        val size = width * height
        val quarter = size / 4
        input.copyInto(output, 0, 0, size)
        for (n in 0 until quarter) {
            output[size + n * 2] = input[size + quarter + n]
            output[size + n * 2 + 1] = input[size + n]
        }
        return output
    }
}

// --- AVC ENCODER ---
class AvcEncoder(
    val width: Int,   
    val height: Int,  
    private val outputCallback: (ByteArray) -> Unit
) {
    private val mediaCodec: MediaCodec
    private val bufferInfo = MediaCodec.BufferInfo()
    @Volatile private var isRunning = true
    private var frameIndex = 0L
    private var configByte: ByteArray? = null
    private var framesEncoded = 0L

    init {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar)
            setInteger(MediaFormat.KEY_BIT_RATE, 750_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, TARGET_FPS)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
        }
        mediaCodec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        mediaCodec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        mediaCodec.start()
        Log.d(TAG, "AvcEncoder: Initialized ($width x $height @ $TARGET_FPS FPS)")
    }

    fun isSize(w: Int, h: Int): Boolean {
        return this.width == w && this.height == h
    }

    fun encode(i420Data: ByteArray) {
        if (!isRunning) return

        val requiredSize = (width * height * 1.5).toInt()
        if (i420Data.size < requiredSize) {
            Log.w(TAG, "AvcEncoder: Input data too small")
            return
        }

        try {
            val inputBufferIndex = mediaCodec.dequeueInputBuffer(5000)
            if (inputBufferIndex >= 0) {
                val inputBuffer = mediaCodec.getInputBuffer(inputBufferIndex)
                
                try {
                    inputBuffer?.clear()
                    val nv12 = i420ToNv12(i420Data, width, height)
                    inputBuffer?.put(nv12)
                    val pts = frameIndex * 1_000_000 / TARGET_FPS
                    mediaCodec.queueInputBuffer(inputBufferIndex, 0, nv12.size, pts, 0)
                    frameIndex++
                } catch (e: Exception) {
                    Log.e(TAG, "AvcEncoder: Conversion Error", e)
                    mediaCodec.queueInputBuffer(inputBufferIndex, 0, 0, 0, 0)
                }
            }
            
            drainOutput()
            
        } catch (e: Exception) {
            Log.e(TAG, "AvcEncoder: Encoding error", e)
        }
    }

    private fun drainOutput() {
        var outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, 0)
        while (outputBufferIndex >= 0) {
            val outputBuffer = mediaCodec.getOutputBuffer(outputBufferIndex)
            val outData = ByteArray(bufferInfo.size)
            outputBuffer?.get(outData)

            if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                configByte = outData
            } else if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0) {
                if (configByte != null) {
                    val combined = ByteArray(configByte!!.size + outData.size)
                    System.arraycopy(configByte!!, 0, combined, 0, configByte!!.size)
                    System.arraycopy(outData, 0, combined, configByte!!.size, outData.size)
                    outputCallback(combined)
                } else {
                    outputCallback(outData)
                }
            } else {
                outputCallback(outData)
            }

            framesEncoded++
            if (framesEncoded % 25L == 0L) {
               Log.d(TAG, "AvcEncoder: Encoded $framesEncoded frames") 
            }
            
            mediaCodec.releaseOutputBuffer(outputBufferIndex, false)
            outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, 0)
        }
    }

    fun close() {
        isRunning = false
        try {
            mediaCodec.stop()
            mediaCodec.release()
            Log.d(TAG, "AvcEncoder: Closed (Total frames: $framesEncoded)")
        } catch (e: Exception) {
            Log.e(TAG, "AvcEncoder: Close error", e)
        }
    }

    private fun i420ToNv12(input: ByteArray, width: Int, height: Int): ByteArray {
        val frameSize = width * height
        val qFrameSize = frameSize / 4
        val output = ByteArray(input.size)
        System.arraycopy(input, 0, output, 0, frameSize) 
        val uOffset = frameSize
        val vOffset = frameSize + qFrameSize
        for (i in 0 until qFrameSize) {
            output[frameSize + i * 2] = input[uOffset + i]
            output[frameSize + i * 2 + 1] = input[vOffset + i]
        }
        return output
    }
}

// --- SERVER CLIENT ---
object ServerClient {
    private var webSocket: WebSocket? = null
    private val framesSent = AtomicLong(0)
    private val framesDropped = AtomicLong(0)
    private val queuedMessages = AtomicInteger(0)
    
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(0, TimeUnit.MILLISECONDS)
        .build()

    fun connect(url: String) {
        if (webSocket != null) close()
        
        Log.d(TAG, "ServerClient: Connecting to $url")
        framesSent.set(0)
        framesDropped.set(0)
        queuedMessages.set(0)
        
        val request = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
                Log.d(TAG, "ServerClient: CONNECTED")
            }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: okhttp3.Response?) {
                Log.e(TAG, "ServerClient: CONNECTION FAILED", t)
            }
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "ServerClient: CLOSED ($reason)")
            }
        })
    }

    fun sendFrame(data: ByteArray): Boolean {
        val ws = webSocket ?: return false
        
        val currentQueued = queuedMessages.get()
        if (currentQueued > 10) {
            val dropped = framesDropped.incrementAndGet()
            if (dropped % 5L == 1L) {
                Log.w(TAG, "ServerClient: Dropping frame (Queue: $currentQueued, Total Dropped: $dropped)")
            }
            return false
        }
        
        try {
            queuedMessages.incrementAndGet()
            val success = ws.send(data.toByteString())
            
            if (success) {
                val sent = framesSent.incrementAndGet()
                queuedMessages.decrementAndGet()
                
                if (sent % 25L == 0L) {
                    Log.d(TAG, "ServerClient: Sent $sent frames (Dropped: ${framesDropped.get()}, Queued: ${queuedMessages.get()})")
                }
                return true
            } else {
                queuedMessages.decrementAndGet()
                framesDropped.incrementAndGet()
                return false
            }
        } catch (e: Exception) {
            queuedMessages.decrementAndGet()
            framesDropped.incrementAndGet()
            Log.e(TAG, "ServerClient: Send error", e)
            return false
        }
    }

    fun close() {
        try {
            Log.d(TAG, "ServerClient: Closing (Sent: ${framesSent.get()}, Dropped: ${framesDropped.get()})")
            webSocket?.close(1000, "Stream stopped")
            webSocket = null
        } catch (e: Exception) {
            Log.e(TAG, "ServerClient: Close error", e)
        }
    }
}
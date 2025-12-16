package com.seenitt

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.meta.wearable.dat.camera.StreamSession
import com.meta.wearable.dat.camera.startStreamSession
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.VideoQuality
import com.meta.wearable.dat.camera.types.VideoFrame
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.AutoDeviceSelector
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import com.meta.wearable.dat.core.types.RegistrationState
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream

class MainActivity : ComponentActivity() {

    private lateinit var btnConnect: Button
    private lateinit var btnStream: Button
    private lateinit var tvStatus: TextView
    private lateinit var ivPreview: ImageView

    private var streamSession: StreamSession? = null
    private var videoJob: Job? = null

    private val permissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        if (permissions.all { it.value }) {
            initializeWearables()
        } else {
            Toast.makeText(this, "Permissions required", Toast.LENGTH_SHORT).show()
        }
    }

    private val wearablesPermissionLauncher = registerForActivityResult(
        Wearables.RequestPermissionContract()
    ) { result ->
        if (result == PermissionStatus.Granted) {
            // Workaround for "Video occasionally freezes on second or later runs"
            // from Known Issues: Add short delay after permission check/grant
            lifecycleScope.launch {
                kotlinx.coroutines.delay(1000)
                startStreaming()
            }
        } else {
            Toast.makeText(this, "Camera permission denied on device", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnConnect = findViewById(R.id.btn_connect)
        btnStream = findViewById(R.id.btn_stream)
        tvStatus = findViewById(R.id.tv_status)
        ivPreview = findViewById(R.id.iv_preview)

        btnConnect.setOnClickListener {
            Wearables.startRegistration(this)
        }

        btnStream.setOnClickListener {
            checkAndStartStreaming()
        }

        checkAndroidPermissions()
    }

    private fun checkAndroidPermissions() {
        val permissions = arrayOf(
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.INTERNET
        )
        if (permissions.all { ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED }) {
            initializeWearables()
        } else {
            permissionsLauncher.launch(permissions)
        }
    }

    private fun initializeWearables() {
        Wearables.initialize(this)
        
        lifecycleScope.launch {
            Wearables.registrationState.collect { state ->
                when (state) {
                    is RegistrationState.Registered -> {
                        tvStatus.text = "Status: Registered"
                        btnConnect.isEnabled = false
                        btnStream.isEnabled = true
                    }
                    else -> {
                        tvStatus.text = "Status: ${state::class.simpleName}"
                        btnConnect.isEnabled = true
                        btnStream.isEnabled = false
                    }
                }
            }
        }
    }

    private fun checkAndStartStreaming() {
        lifecycleScope.launch {
            val status = Wearables.checkPermissionStatus(Permission.CAMERA)
            if (status == PermissionStatus.Granted) {
                // Workaround for "Video occasionally freezes on second or later runs"
                // from Known Issues: Add short delay after permission check/grant
                kotlinx.coroutines.delay(1000)
                startStreaming()
            } else {
                wearablesPermissionLauncher.launch(Permission.CAMERA)
            }
        }
    }

    private fun startStreaming() {
        if (streamSession != null) {
            return
        }

        lifecycleScope.launch {
            try {
                val session = Wearables.startStreamSession(
                    this@MainActivity,
                    AutoDeviceSelector(),
                    StreamConfiguration(VideoQuality.MEDIUM, 24)
                )
                streamSession = session
                btnStream.text = "Stop Streaming"
                btnStream.setOnClickListener { stopStreaming() }

                videoJob = launch {
                    session.videoStream.collect { frame ->
                        val bitmap = processFrame(frame)
                        if (bitmap != null) {
                            runOnUiThread {
                                ivPreview.setImageBitmap(bitmap)
                            }
                            // Send to server
                            ServerClient.sendFrame(bitmap)
                        }
                    }
                }
                
                // Monitor session state
                launch {
                    session.state.collect { state ->
                        tvStatus.text = "Stream State: ${state::class.simpleName}"
                        if (state == com.meta.wearable.dat.camera.types.StreamSessionState.STOPPED) {
                            stopStreaming()
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Error starting stream", e
        videoJob?.cancel()
        streamSession?.close()
        streamSession = null
        btnStream.text = "Start Streaming"
        btnStream.setOnClickListener { checkAndStartStreaming() }
        ivPreview.setImageDrawable(null)
    }

    private fun processFrame(videoFrame: VideoFrame): Bitmap? {
        try {
            val buffer = videoFrame.buffer
            val width = videoFrame.width
            val height = videoFrame.height
            
            val dataSize = buffer.remaining()
            val byteArray = ByteArray(dataSize)
            
            val originalPosition = buffer.position()
            buffer.get(byteArray)
            buffer.position(originalPosition)

            // Check if we have enough data for I420 (Y + U + V)
            // I420 size = width * height * 1.5
            val expectedSize = (width * height * 1.5).toInt()
            if (byteArray.size < expectedSize) {
                return null
            }

            val nv21 = convertI420toNV21(byteArray, width, height)
            val image = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            image.compressToJpeg(Rect(0, 0, width, height), 80, out)
            val imageBytes = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        } catch (e: Exception) {
            Log.e("MainActivity", "Error processing frame", e)
            return null
        }
    }

    private fun convertI420toNV21(input: ByteArray, width: Int, height: Int): ByteArray {
        val output = ByteArray(input.size)
        val size = width * height
        val quarter = size / 4

        input.copyInto(output, 0, 0, size) // Y is the same

        for (n in 0 until quarter) {
            output[size + n * 2] = input[size + quarter + n] // V first
            output[size + n * 2 + 1] = input[size + n] // U second
        }
        return output
    }
}

object ServerClient {
    fun sendFrame(bitmap: Bitmap) {
        // TODO: Implement server communication
        // Example: Convert bitmap to Base64 or bytes and send via WebSocket or HTTP
    }
}

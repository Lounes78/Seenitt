# Seenitt Debug Commands

## ADB Commands to Monitor Logs

### 1. **Monitor ALL Seenitt App Logs**
```bash
adb logcat -s MainActivity MetaWearablesConfig DAT:CORE
```

### 2. **Monitor ONLY Critical Registration Flow** (Recommended)
```bash
adb logcat | grep -E "MainActivity|RegistrationState|DAT:CORE|MetaWearables"
```

### 3. **Monitor with Timestamps**
```bash
adb logcat -v time -s MainActivity MetaWearablesConfig DAT:CORE
```

### 4. **Clear Logs and Start Fresh**
```bash
adb logcat -c
adb logcat -s MainActivity MetaWearablesConfig DAT:CORE:* DAT:BLE:*
```

### 5. **Save Logs to File**
```bash
adb logcat -s MainActivity MetaWearablesConfig DAT:CORE > seenitt_debug.log
```

### 6. **Monitor Bluetooth Device Detection**
```bash
adb logcat | grep -E "MainActivity|BluetoothDeviceDetection|DAT:CORE:BluetoothLE"
```

## What to Look For

### ✅ **Good Indicators:**
- `=== Initializing Wearables SDK ===`
- `=== Connect Button Clicked - Starting Registration ===`
- `Registration State Changed: InProgress`
- `Registration State Changed: Registered`
- `Device REGISTERED successfully!`

### ⚠️ **Warning Indicators (Expected in Developer Mode):**
- `CLIENT_TOKEN not found in manifest metadata` ← Expected, ignore
- `ANALYTICS_OPT_OUT not found in manifest metadata` ← Expected, ignore

### ❌ **Problem Indicators:**
- Stuck in `InProgress` state for more than 30 seconds
- `BluetoothDeviceDetectionProvider already started` (might indicate re-initialization)
- Any error messages with stack traces
- `Registration State Changed: Unregistered` (after attempting to connect)

## Rebuild and Test Steps

1. **Clean and Rebuild:**
   ```bash
   cd c:\Users\az03732\AndroidStudioProjects\Seenitt
   .\gradlew clean
   .\gradlew assembleDebug
   ```

2. **Install App:**
   ```bash
   adb install -r app\build\outputs\apk\debug\app-debug.apk
   ```

3. **Start Monitoring:**
   ```bash
   adb logcat -c
   adb logcat -s MainActivity MetaWearablesConfig DAT:CORE
   ```

4. **Test the App:**
   - Launch Seenitt
   - Click "Connect" button
   - Watch the logs

## Expected Log Flow

```
MainActivity: === Initializing Wearables SDK ===
MainActivity: === Registration State Changed: Unregistered ===
MainActivity: Device UNREGISTERED - Ready to connect
[User clicks Connect button]
MainActivity: === Connect Button Clicked - Starting Registration ===
MainActivity: === Registration State Changed: InProgress ===
MainActivity: Registration IN PROGRESS - Searching for device...
[Meta AI app should open for confirmation]
[After user confirms in Meta AI]
MainActivity: === Registration State Changed: Registered ===
MainActivity: Device REGISTERED successfully!
```

## Common Issues

### Issue: Registration Returns "FAILED_TO_REGISTER"
**This is your current issue!**

**Root Cause:** Developer Mode requires proper setup in Meta View app

**Required Steps:**
1. **Pair glasses in Meta View app first** (not Meta AI)
   - Open Meta View app
   - Go through the pairing process
   - Ensure glasses are connected

2. **Enable Developer Mode in Meta View:**
   - Open Meta View app
   - Go to Settings
   - Look for "Developer" or "Advanced" settings
   - Enable "Developer Mode" or "Third-party apps"
   - You may need to enable "USB Debugging" on glasses

3. **Grant permissions in Meta View:**
   - Your app needs to be authorized
   - In Meta View settings, check for "Connected Apps" or "Third-party apps"
   - Ensure Seenitt is allowed

4. **Verify glasses are connected:**
   ```bash
   # Check Bluetooth devices
   adb shell dumpsys bluetooth_manager | grep "RayBan\|Meta"
   
   # Check if Meta View sees the glasses
   adb shell dumpsys activity activities | grep meta
   ```

**Debug Commands:**
```bash
# Check if Meta View is installed (required before Meta AI works)
adb shell pm list packages | grep meta

# Check Bluetooth status
adb shell dumpsys bluetooth_manager

# Check for connected Bluetooth devices
adb shell dumpsys bluetooth_manager | grep "Device"
```

### Issue: Meta AI Opens but Shows Error (Your Current Issue)
**This is what "FAILED_TO_REGISTER" means**

**Solution - Follow this order:**
1. ✅ **Meta View app installed and updated**
2. ✅ **Glasses paired in Meta View app**
3. ✅ **Developer Mode enabled in Meta View settings**
4. ✅ **APPLICATION_ID set to "0"** (already done)
5. ⚠️ **Grant third-party app access in Meta View**
6. ⚠️ **Try registering again in your app**

**Common Fixes:**
- Unpair and re-pair glasses in Meta View
- Clear Meta AI app data: `adb shell pm clear com.facebook.stella`
- Clear Meta View app data: `adb shell pm clear com.meta.viewapp`
- Restart phone with glasses connected
- Update Meta View and Meta AI to latest versions

## Additional Diagnostic Info

```bash
# Get all package info
adb shell dumpsys package com.seenitt

# Get app permissions
adb shell dumpsys package com.seenitt | grep permission

# Monitor all Bluetooth LE activity
adb logcat -s BluetoothGatt:* BluetoothAdapter:*
usbipd list

usbipd bind --busid 2-1

Note that as long as the USB device is attached to WSL, it cannot be used by Windows. Once attached to WSL, the USB device can be used by any distribution running as WSL 2. Verify that the device is attached using usbipd list

usbipd attach --wsl --busid 2-1

Open Ubuntu (or your preferred WSL command line) and list the attached USB devices using the command:
lsusb

Once you are done using the device in WSL, you can either physically disconnect the USB device or run this command from PowerShell:
usbipd detach --busid 2-1
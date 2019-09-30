old code:  
```
from picamera import PiCamera

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
#24 may be wrong


class button():
    def __init__(self,pid):
        self.pressed = False
        self.pid = pid
    def check_state(self):
        return GPIO.input(self.pid)
    def one_time_press(self):
        state = self.check_state()
        if state == True:
            self.pressed = False
            return False
        else:
            if self.pressed == False:
                self.pressed = True
                return True
            else:
                return False

print('welcome to jimmy and summer\'s camera')
#camera = PiCamera()
#camera.start_preview()
shutter = button(18)
mode_button = button(23)
mode = 0 #0 1 2
mode_doc = ['camera', 'recording', 'black and white']
recording = False

#add start preview in various places
while True:
    if mode_button.one_time_press():
        mode = mode + 1
        mode = mode % 3
        print('we are now in '+ mode_doc[mode]+ ' mode')
    if mode == 0:
        if shutter.one_time_press():
            print('take a picture')
    elif mode == 1: 
        if shutter.one_time_press():
            if recording:
                print('recording end')
                recording = False
            else:
                print('start recording...')
                recording = True
    else:
        if shutter.one_time_press():
            print('take a black and white picture')
        
    time.sleep(0.01)
```





new code:  
```
import os
from picamera import PiCamera

import RPi.GPIO as GPIO
import time

"""class button():
    def __init__(self,pid):
        self.pressed = False
        self.pid = pid
    def check_state(self):
        return GPIO.input(self.pid)
    def one_time_press(self):
        state = self.check_state()
        if state == True:
            self.pressed = False
            return False
        else:
            if self.pressed == False:
                self.pressed = True
                return True
            else:
                return False
"""
def modeChange(channe):
    global mode
    global recording
    if recording:
        print('is recording now... if you want to change mode, please stop record')
    else:
        mode = (mode + 1) % 3
        print('we are now in '+ mode_doc[mode]+ ' mode')

def shutterChange(channel):
    global mode
    global recording
    if mode == 0:
        print('take a picture')
    elif mode == 1:
        if recording:
            print('recording end')
        else:
            print('start recording...')
        recording = not recording
    else:
        print('take a black and white picture')

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    #24 may be wrong
    print('welcome to jimmy and summer\'s camera')
    #camera = PiCamera()
    #camera.start_preview()
    #shutter = button(18)
    #mode_button = button(23)
    global mode = 0 #0 1 2
    global mode_doc = ['camera', 'recording', 'black and white']
    global recording = False
    GPIO.add_event_detect(18, GPIO.FALLING, callback = shutterChange, bouncetime = 100)
    GPIO.add_event_detect(23, GPIO.FALLING, callback = modeChange, bouncetime = 100)

def destory():
    GPIO.remove_event_detect(18)
    GPIO.remove_event_detect(23)
    GPIO.cleanup()

if __name__ == '__main__':
    try:
        setup()
    except KeyboardInterrupt:
        destory()
    destory()
    #add start preview in various places
```

final
```
from picamera import PiCamera

import RPi.GPIO as GPIO
import time

def modeChange(channe):
    global mode
    global recording
    if recording:
        print('is recording now... if you want to change mode, please stop record')
    else:
        mode = (mode + 1) % 3
        print('we are now in '+ mode_doc[mode]+ ' mode')

def shutterChange(channel):
    global mode
    global recording
    global serial_pic, serial_rec
    if mode == 0:
        serial_pic = serial_pic + 1
        caption = 'DCF_'+'0'*(4-len(str(serial_pic)))+str(serial_pic)
        path = '/home/pi/Camera/DCIM/100JLIFE/'+caption+'.jpg'
        camera.capture(path)
    
    elif mode == 1:
        if recording:
            camera.stop_recording()
        else:
            serial_rec = serial_rec + 1
            caption = 'DSC_'+'0'*(4-len(str(serial_rec)))+str(serial_rec)
            path = '/home/pi/Camera/499VIDEO/'+caption+'.jpg'
            camera.start_recording(path)
        recording = not recording
    else:
        serial_pic = serial_pic + 1
        caption = 'DCF_'+'0'*(4-len(str(serial_pic)))+str(serial_pic)
        path = '/home/pi/Camera/DCIM/100JLIFE/'+caption+'.jpg'
        for i in range(3,0,-1):
            print('Countdown: ',i)
            time.sleep(1)
        camera.capture(path)        
print('take a black and white picture')

def setup():
    if not os.path.exists('home/pi/Camera'):
        os.makedirs('home/pi/Camera')
    if not os.path.exists('home/pi/Camera/DCIM'):
        os.makedirs('home/pi/Camera/DCIM')
    if not os.path.exists('home/pi/Camera/DCIM/100JLIFE'):
        os.makedirs('home/pi/Camera/DCIM/100JLIFE')
    if not os.path.exists('home/pi/Camera/DCIM/499VIDEO'):
        os.makedirs('home/pi/Camera/DCIM/499VIDEO')        
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    print('welcome to jimmy and summer\'s camera')
    global camera
    camera = PiCamera()
    camera.start_preview()
    time.sleep(3)   #to give the sensor time to set its light levels.
    global serial_pic, serial_rec, mode, mode_doc, recording
    serial_pic = 0
    serial_rec = 0
    mode = 0 #0 1 2
    mode_doc = ['camera', 'recording', 'black and white']
    recording = False
    GPIO.add_event_detect(18, GPIO.FALLING, callback = shutterChange, bouncetime = 100)
    GPIO.add_event_detect(23, GPIO.FALLING, callback = modeChange, bouncetime = 100)

def destory():
    GPIO.remove_event_detect(18)
    GPIO.remove_event_detect(23)
    GPIO.cleanup()

if __name__ == '__main__':
    try:
        setup()
    except KeyboardInterrupt:
        destory()
    destory()
    time.sleep(1e7)  # 1e7 seconds ~= 1 year
```

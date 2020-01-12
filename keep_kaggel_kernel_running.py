import pyautogui as auto
import time
current_x,current_y=auto.position()
print('current:'+str(current_x)+','+str(current_y))
step=0
print('Start')
while True:
    time.sleep(6)
    print(time.ctime())
    print('move')
    #auto.moveRel(xOffset=200,yOffset=100,duration=0.0,tween=auto.linear)
    current_x,current_y=auto.position()
    print('current:'+str(current_x)+','+str(current_y))
    print('scroll')
    auto.scroll(clicks=10)
    #auto.click(button='left')
    time.sleep(6)
    #auto.moveRel(xOffset=-200,yOffset=-100,duration=0.0,tween=auto.linear)
    current_x,current_y=auto.position()
    print('current:'+str(current_x)+','+str(current_y))
    step+=1
    print('step:'+str(step))
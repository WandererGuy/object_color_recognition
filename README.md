prepare env like in prepare.md 

Usage: run
create_rgb.py -> create rgb representation of a hue range 
infer.py -> is for infer 
main.py -> create fastaspi server for infer through API 


Description:
this code is object color recognition 
each object  has predominant hue color range 
there is 12 hue color range on color wheel theory and 1 more range of black_grey_white

Hue is better than BGR since it tells true color 
though weakness in this code , is 
if colorful light shine on object , then it might detect the color of that light
if too much shadow onto object, it might detect black
if color of object too fade or so much light onto object, it might detect white
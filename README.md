# mudra
tools for 3d imaging in the context of the MUltimodal Data Registration Algorithm

# Scalar Field Intepolator

Usage : python sfi_viewer.py input_file.npy

Keys list :
- '!': # Start / Stop Cropping
- '@': # Load a Sampling Arrow (then clic)
- "#": #  Start/Stop Interpolation Plane (IP)
- 'd': # Show / Hide  IP handle
- '+': Tranlate + IP along it normal
- '-': Tranlate - IP along it normal
- 'c': Show / Hide  Volume Rendering (VR)
- "r": Emply the sampling point list
- 'l': List tje Sampling point list
- '$': Open matplotlib imshowe with the IP image
- '^': Save Screenshot
   
avec "!" tu limite le volume rendering avec une crop box "!" a nouveau pour la faire disparaitre
pour cacher/faire reparaitre le volume rendering appui sur "c"
avec "#" tu fait apparaitre un plan d'interpolation : pour le déplacer lit ce qui suit :


By grabbing the one of the four handles (use the left mouse button), the plane can be resized. 
By grabbing the plane itself, the entire plane can be arbitrarily translated. 
Pressing CTRL while grabbing the plane will spin the plane around the normal. 
If you select the normal vector, the plane can be arbitrarily rotated. 
Selecting any part of the widget with the middle mouse button enables translation of the plane along its normal. 
(Once selected using middle mouse, moving the mouse in the direction of the normal translates the plane in the direction of the normal; 
moving in the direction opposite the normal translates the plane in the direction opposite the normal.) 
Scaling (about the center of the plane) is achieved by using the right mouse button. 
By moving the mouse "up" the render window the plane will be made bigger; 
by moving "down" the render window the widget will be made smaller


pour bloquer le plan d'interpolation appui sur "d" 

une fois qu'il ya un plan de défini sur peut appuyer sur "@" ca permet de lancer une fléchette sur le plan 
(et de définir une coordonnée dans les 3 dimensions).
il faut presser "@" pour recharger une fléchette a chaque fois et cliquer sur le plan.
en pressant sur "l"(L minuscule) le programme va te donner la liste des coordonnées entières 
les plus proche associées aux intensitée du champs scalaire en ces points

presse "$" (toujours en association avec le plan d'interpolation) pour ouvrir un imshow matplotlib de l'image interpolée sur le plan.

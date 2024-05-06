import cv2 as cv
import numpy as np
'''
https://amroamroamro.github.io/mexopencv/matlab/cv.solveP3P.html
'''

class p3p_solver:
    def __init__(self, input_array, camera_intrinsic):
        self.input_array = input_array
        self.intrinsic = camera_intrinsic
    
    def p3p_solver(self, object_point, image_point):
        '''
        input : 
            object_point { [x1,y1,z1], [x2,y2,z2], [x3,y3,z3] }
            image_point  { [u1,v1], [u2,v2], [u3,v3] }
            camera_matrix A = [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
        '''
        rotation_vector, transition_vector, num_of_solutions = cv.solvePnPRANSAC(object_point, image_point, self.intrinsic)

        return rotation_vector, transition_vector
 
    def p3p_select_point(self):
        pass

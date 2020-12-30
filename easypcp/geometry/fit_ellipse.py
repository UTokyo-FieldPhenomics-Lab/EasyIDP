import numpy as np
from numpy.linalg import eig, inv

'''
Source code ref: http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
'''
def fit_ellipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D =  np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    return V[:, n]

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_axis_length(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(np.abs(up/down1))
    res2=np.sqrt(np.abs(up/down2))
    return np.array([res1, res2])

def ellipse_angle_of_rotation2(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

if __name__ == '__main__':
    x = np.array([5727.53135, 7147.62235, 10330.93573, 8711.17228, 7630.40262,
                  4777.24983, 4828.27655, 9449.94416, 5203.81323, 6299.44811,
                  6494.21906])

    y = np.array([67157.77567, 66568.50068, 55922.56257, 54887.47348,
                  65150.14064, 66529.91705, 65934.25548, 55351.57612,
                  63123.5103, 67181.141725, 56321.36025])

    a = fit_ellipse(x, y)
    # > array([-1.10182740e-09, -5.13462963e-10, -1.93678403e-10,  4.74839785e-05,
    #         2.73250882e-05, -9.99999998e-01])

    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation2(a)
    axes = ellipse_axis_length(a)

    half_h, half_v = axes
    rot_angle =  phi * 180 / np.pi + 90

    '''
    import matplotlib.pyplot as plt
    
    ell = Ellipse(center, 2 * a, 2 * b,  phi * 180 / np.pi + 90 ) 

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    scat = plt.scatter(x, y, c = "r")
    plt.show()
    '''
from aerosandbox.geometry.polygon import *
import aerosandbox.numpy as np
import pytest

# Unit square TODO: Check bahavior when flipped
xy = np.flip(np.array([
        [0, 0], [1, 0],
        [1, 1], [0, 1],
        [0, 0]
        ]), axis=0)

x = xy[:, 0]
y = xy[:, 1]

def test_area1():
    poly = Polygon(x, y)
    
    assert all(poly.x() == x)
    assert all(poly.y() == y)
    assert poly.area() == 1

def test_area2():
    poly = Polygon(xy)
    
    assert all(poly.x() == x)
    assert all(poly.y() == y)
    assert poly.area() == 1
    
def test_area3():
    x = np.vstack([xy[:, 0]]*100)
    y = np.vstack([xy[:, 1]]*100)
    
    poly = Polygon(x, y)
    
    assert np.all(poly.x() == x)
    assert np.all(poly.y() == y)
    assert np.all(poly.area() == np.ones((100,)))
    

if __name__ == '__main__':
    pytest.main()
    
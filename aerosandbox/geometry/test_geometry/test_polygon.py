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
    
    
def contains_point1():
    poly = Polygon(xy)
    
    assert poly.contains_points(0.5, 0.5)
    
    
def centroid1():
    poly = Polygon(xy)
    
    assert poly.centroid() == np.array([0.5, 0.5])
    
    
def I_J():
    poly = Polygon(xy)
    
    assert np.isclose(poly.Ixx(), 0.08333333333333331)
    assert np.isclose(poly.Iyy(), 0.08333333333333331)
    assert np.isclose(poly.Ixy(), 0)
    assert np.isclose(poly.J(), 0.16666666666666663)
    

if __name__ == '__main__':
    pytest.main()
    
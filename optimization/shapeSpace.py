# shapeSpace.py
import torch


def getInnerProduct(FirP, SecP):
    """计算两个向量之间的点积"""
    # InnerProd = 0.0
    # for i in range(FirP.shape[0]):
    #     InnerProd += FirP[i] * SecP[i]
    InnerProd = torch.matmul(FirP, SecP.T)
    return InnerProd

def getShapeSpaceDistance(FirP, SecP):
    """两个形状之间的距离"""
    # InnProd = getInnerProduct(FirP, SecP)
    InnProd = torch.matmul(FirP, SecP.T)

    InnProd = torch.clamp(InnProd, -1, 1)

    dist = torch.arccos(InnProd)
    return dist

def getShapeSpaceDistanc_proj(FirP, SecP):
    """对于还未投影的输入，先投影，再计算两个形状之间的距离"""
    # InnProd = getInnerProduct(FirP, SecP)
    p1 = project(FirP)
    p2 = project(SecP)
    return getShapeSpaceDistance(p1, p2)



def project(gen_logits):
    '''
    gen_logits: (batch_size, num_features)
    Args:
        gen_logits:

    Returns:
        gen_logits: projected to the shape space (batch_size, num_features)
    '''

    gen_logits = gen_logits.reshape(gen_logits.shape[0], -1, 2)
    # gen_logits = gen_logits.type(torch.DoubleTensor)
    mean = torch.mean(gen_logits, dim=1)
    #for i in range(gen_logits.shape[0]):
    #    gen_logits[i] = gen_logits[i] - mean[i]

    mean = mean.repeat_interleave(gen_logits.shape[1], dim=0).reshape(gen_logits.shape[0], gen_logits.shape[1], gen_logits.shape[2])
    gen_logits = gen_logits - mean

    # gen_logits = gen_logits - mean.unsqueeze(1)
    gen_logits = gen_logits.reshape(gen_logits.shape[0], -1)
    norm = torch.linalg.norm(gen_logits, dim=1, keepdim=True)
    gen_logits = gen_logits / norm
    # for i in range(gen_logits.shape[0]):
    #     gen_logits[i].data = Kendall.project(gen_logits[i])
    # gen_logits = gen_logits.type(torch.FloatTensor)
    # gen_logits = gen_logits.reshape(gen_logits.shape[0], -1)

    return gen_logits


class ShapeInterpolator:
    """
    This class implements a routine for 3D shape estimation from 2D landmarks
    presented in [1].

    The estimation problems is regularized by employing prior knowledge learned from
    examples of the target's object class. In particular, the 3D shape of the target
    is estimated by interpolating through a set of pre-defined 3D shapes, called
    training shapes.

    Parameters
    ----------
    W : array-like, shape=[k, 2]
        Projected 2D landmark positions of target shape.
    B : array-like, shape=[N, k, 3]
        Set of N training shapes in 3D.

    References
    ----------
    .. [1] M. Paskin, D. Baum, Mason N. Dean, Christoph von Tycowicz (2022).
    A Kendall Shape Space Approach to 3D Shape Estimation from 2D Landmarks.
    In: European Conference on Computer Vision. Springer

    """

    def __init__(self, shapes, weights):
        # self.W = W

        self.shapes = shapes
        self.numShapes = shapes.shape[0]

        self.weights = weights
        self.frechetMean = None

    def mean(self, B, c):
        # mfd = Kendall(shape=B[0].shape)
        fmean = generate_from_geodesic_surface(B, c)
        fmean = fmean.reshape(-1, 2)
        # print('fmean', fmean.shape)
        # return mfd.wellpos(B[0], fmean)  # fmean aligned to B[0]
        return fmean

    def generate(self, tol=1e-6, verbose=0):
        """
        Alternating optimization for 3D-from-2D problem.

        Parameters
        ----------
        tol : array-like, shape=[N, ...]
            Tolerance for convergence check.
            Optional, default: 1e-6.
        verbose : int
            Verbosity level: 0 is silent, 2 is most verbose.
            Optional, default: 0

        Returns
        -------
        mean : array-like, shape=[N, ...]
            Weighted Frechet mean.
        """
        # print(self.weights)

        # self.rotation = jnp.asarray(V.random_point())
        mean = self.mean(self.shapes, self.weights)

        self.frechetMean = mean
        return self.frechetMean

def geopoint(FirP, SecP, angle_s): #w0, w1, temps
    """测地线公式"""

    # NewP = FirP.copy()

    dist = getShapeSpaceDistance(FirP, SecP)
    angle_s = dist * angle_s

    cosMidDis = torch.cos(angle_s)
    sinMidDis = torch.sin(angle_s)

    cosDist = torch.cos(dist)
    sinDist = torch.sin(dist)

    if dist < 0.00005:
        return FirP


    NewP = cosMidDis * FirP + \
            sinMidDis * ((SecP - FirP * cosDist) / sinDist)

    # print()

    return NewP

def generate_from_geodesic_surface(shapes, weights):
    # Frechet mean of a pointset on a manifold
    # mfd: manifold
    # pointset: pointset on the manifold
    # weights: weights of the points in the pointset
    # returns: Frechet mean of the pointset
    """
    Estimate weighted Frechet mean via recursive method.

    Parameters
    ----------
    mfd : Manifold
        Shape space.
    pointset : array-like, shape=[N, ...]
        Collection of sample points on mfd.
    weights : array-like, shape=[N]
        Weights.

    Returns
    -------
    mean : array-like, shape=[N, ...]
        Weighted Frechet mean.
    """
    # number of points
    # fmean = frechetMean(mfd, B, c)
    num = weights.shape[0]

    # weights = weights / weights.sum()  # sum(weights) = 1
    shapes = shapes.reshape(num, -1)

    # print('weights', weights.shape)
    # idx = jnp.argsort(-weights)  # 从大到小排序
    # weights = weights[idx]
    t = weights / (torch.cumsum(weights, dim=0) + 1e-8)
    # print(t.shape)
    # cumsum 按行累加

    # loop = lambda i, mean: mfd.connec.geopoint(mean, shapes[i], t[i])
    loop = lambda i, mean: geopoint(mean, shapes[i], t[i])
    # geopoint算的是geodesic distance

    def fori_loop(lower, upper, body_fun, init_val):
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val

    generated_feature = fori_loop(1, num, loop, shapes[0])


    return generated_feature
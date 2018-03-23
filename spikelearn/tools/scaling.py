# find eigenvalue S, and corresponding eigenvector X, using Reighley quotient R(A,B)= x'Ax/x'Bx = X' S X / X'X
# Example:
# A = (x1 - x2)' * ( x1 - x2); % L2 normal, x's dimension is nxp, n = number of observations, p = dimensionality.
# B = cov([x1; x2]);
# [S,X] = ReighleyQuotient(A,B,'min',1)
#
# function [S,X]=ReighleyQuotient(A,B,ORDER,ORTH)
# %Reighley quotient R(A,B)= x'Ax/x'Bx = X' S X / X'X
# %find eigenvalue S, and corresponding eigenvector X
# %ORDER='min', minimim first, ascend order of S, ORDER='max', descend order of S
# %ORTH=1, orthogonalize eigenvectors.
# % by Jing Wang, jingwang.physics@gmail.com, 05/23/3017
#
# if ishermitian(A) && ishermitian(B)
#     [~,Sb,Vb]=svd(B);
#     %C=Vb*sqrt(Sb);
#     sbinv=1./sqrt(diag(Sb));
#     Sbinv=diag(sbinv);
#     D=Sbinv* Vb'*A*Vb*Sbinv;
#     [~,S,V]=svd(D);
#     X_=Vb*Sbinv*V;
#     IND=[];
#     for i=1:size(X_,2)
#         x=X_(:,i);
#         X_(:,i)=x./norm(x);
#         if norm(x)<1e6
#             IND=[IND;i];
#         end
#     end
#
#     X=X_(:,IND); % remove bad column, if it's rank deficient
#
#     if strcmpi(ORDER, 'min')
#         X=X(:,end:-1:1);
#     end
#
#     if ORTH
#         [X,~]=qr(X);
#     end
#
# else
#     disp('Error!A or B is not Hermitian.')
# end
#
# end

from numpy.linalg import svd, norm, qr
def ReighleyQuotient(A, B, ascending=True, orthogonalize=True, tol=1e6):
    """
    find eigenvalue S, and corresponding eigenvector X, that solves the
    Reighley equation R(A,B) = x'Ax/x'Bx = X' S X / X'X

    Parameters
    ----------
    A : matrix

    B : matrix

    ascending : bools
        Whether to order the eigenpairs by eigenvalue size with minimum first

    orthogonalize : bool, default True
        Whether to QR the matrix and return it orthogonalized.

    tol : float, default 1e6
        The highest value accepted for the norm of a given column.

    Returns
    -------
    S : singular values


    References
    ----------
    From the following article:

    Adapted from the article author's matlab code:
    https://www.mathworks.com/matlabcentral/fileexchange/66547-reighleyquotient-a-b-order-orth-
    """
    assert A == A.H and B == B.H # both are hermitian
    _, s, V = svd(B)
    C = V * np.sqrt(s)
    sinv = (1./np.sqrt(s.diag()))
    D = sinv * V.T * A * V * sinv

    new_s, new_V = svd(D)
    X_ = V*sinv*new_V
    IND = []
    for i in range(X_.shape[1]):
        x = X_[:,i]
        X_[:,i] = x/norm(x)
        if norm(x) < tol:
            IND.append(i)

    X = X_[:,IND]

    if ascending:
        X = X[:,::-1]
    if orthogonalize:
        X, _ = qr(X)

    return new_s, X

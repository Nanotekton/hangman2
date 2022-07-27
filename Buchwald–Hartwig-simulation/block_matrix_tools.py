import numpy as np
from scipy.spatial.distance import cdist

def check_block_shapes(A, B, C):
    a1, a2 = A.shape
    b1, b2 = B.shape
    c1, c2 = C.shape
    msg = 'incorrect shapes A: (%i %i) B: (%i %i) C: (%i %i)'%(a1, a2, b1, b2, c1, c2)
    assert a1==a2 and b1==b2, msg
    assert a1==c2 and b1==c1, msg

def inverse_if_none(M, iM):
    none = type(None)
    if isinstance(iM, none):
        iM = np.linalg.inv(M)
    return iM
        
#TODO: re-use values present in compute_minuend_update
def compute_block_inverse(A, B, C, A_inv=None, B_inv=None):
    check_block_shapes(A, B, C)
    A_inv = inverse_if_none(A, A_inv)
    B_inv = inverse_if_none(B, B_inv)
    
    iA = np.linalg.inv(B_inv - C.dot(A_inv).dot(C.T))
    iA = C.T.dot(iA).dot(C)
    iA = A_inv + A_inv.dot(iA).dot(A_inv)
    
    iC = -B_inv.dot(C).dot(iA)
    
    iB = np.linalg.inv(B - C.dot(A_inv).dot(C.T))
    
    return iA, iB, iC

def compute_minuend_update(A_inv, B, B_inv, C, Kat, Kaq):
    check_block_shapes(A_inv, B_inv, C)
    
    X = Kat.dot(A_inv).dot(C.T) #aq
    Y = C.dot(A_inv).dot(C.T) #qq
    kaq_kqq_inv = Kaq.dot(B_inv) #aq
    kqq_y_x = np.linalg.inv(B-Y).dot(X.T) #qa
    kqq_y = np.linalg.inv(B-Y) #qq

    update = (X*kqq_y_x.T).sum(axis=1) #a
    update -= 2*(kaq_kqq_inv*(X+ Y.dot(kqq_y_x).T)).sum(axis=1)
    update += (Kaq.dot(kqq_y)*Kaq).sum(axis=1)

    return update

def compute_scores(kernel_matrix, noise, train_idx, perceived_optimum_idx):
   K = kernel_matrix
   K += np.identity(len(train_idx))*noise
   sigma = K[0,0]
   B = np.array([[sigma]])
   B_inv = np.array([[1/sigma]])

   if np.array(train_idx).dtype.name=='bool':
      test_idx = ~train_idx
      test_idx = np.where(test_idx)[0]
      train_idx = np.where(train_idx)[0]
   else:
      test_idx = [x for x in np.arange(X.shape[0]) if x not in train_idx]
      train_idx.sort()
      test_idx.sort()
   try:
      query_train_block = K[test_idx,:][:,train_idx]
   except:
      print('WWWWWWWTTTTTTTFFFFFF???????')
      print(train_idx)
      print('WWWWWWWTTTTTTTFFFFFF???????')
      print(test_idx)
      print('WWWWWWWTTTTTTTFFFFFF???????')
      raise

   perceived_rows_block = K[perceived_optimum_idx,:][:,train_idx]
   perceived_query_cross = K[perceived_optimum_idx,:][:, test_idx]
                                 
   Kinv = np.linalg.inv(K[train_idx,:][:,train_idx])
                                       
   #logging.info('Ref from block %s'%str(ref))

   print('Shapes: kernel, Kinv, train, test, perceived:', [np.shape(x) for x in [kernel_matrix, Kinv, train_idx, test_idx, perceived_optimum_idx]])
   print('                                       types:', [type(x) for x in [kernel_matrix, Kinv, train_idx, test_idx, perceived_optimum_idx]])
   print('                                        sums:', [np.sum(x) for x in [kernel_matrix, Kinv, train_idx, test_idx, perceived_optimum_idx]])

   scores = np.zeros(K.shape[0])
   for idx, QT in enumerate(query_train_block):
      QT = QT.reshape(1,-1)
      AQ = perceived_query_cross[:,idx].reshape(-1,1)
      score = compute_minuend_update(Kinv, B, B_inv, QT, perceived_rows_block, AQ)
      scores[test_idx[idx]] = score.sum()

   return scores
                                                                              

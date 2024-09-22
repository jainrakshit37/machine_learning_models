import numpy as np 
 
 
def proj(x, u):   
    u = unit_vec(u) 
    return np.dot(x, u) * u 
 
def unit_vec(x): 
    return x / np.linalg.norm(x) 
 
def gramSchmidt(vectors): 
    if len(vectors) == 0: 
        return [] 
 
    if len(vectors) == 1: 
        return unit_vec(vectors) 
 
    u = vectors[-1] 
    basis = gramSchmidt(vectors[0:-1])   
    w = np.atleast_2d(u - np.sum(proj(u, v) for v in basis))     
    basis = np.append(basis, unit_vec(w), axis=0) 
    return basis 
 
def modifiedGramSchmidt(vectors): 
    vectors = np.atleast_2d(vectors) 
    if len(vectors) == 0: 
        return [] 
    u1 = unit_vec(vectors[0]) 
    if len(vectors) == 1: 
        return u1 
    basis = np.vstack((u1, modifiedGramSchmidt(list(map(lambda v: v - proj(v, u1), vectors[1:]))))) 
    return np.array(basis) 
 
def _is_orthag(vectors): 
    orthag = True 
    vectors = np.atleast_2d(vectors)     
    for vector in vectors: 
        for vector2 in vectors: 
                    if np.array_equal(vector, vector2): 
                        continue 
                    if abs(np.dot(vector, vector2)) > 1e-5: 
                        orthag = False    
    return orthag 
 
 
def test(): 
    #vectors = [[3, 1], [2, 2]]     
    #print("Test 1: Finding orthonormal basis for simple vector space {}".format(vectors)) 
 
    #ospace = gramSchmidt(vectors)     
    #ospace2 = modifiedGramSchmidt(vectors)     
    #samesies = np.array_equal(ospace, ospace2)     
    #print("\nFor vector space provided {}\n\tOrthonormalized basis produced with classical Gram Schmidt process \n{}".format(vectors, ospace))     
    #print( "\n\tOrthonormal basis produced with Modified Gram Schmidt process: \n{}, \nAre the two about equal?? {}".format(ospace2, samesies))     
    #print("\tAre all basis vectors orthagonal to eachother? {}".format(_is_orthag(ospace))) 
 
    vectors = [[3, 13, 2, 5], [1, 1, 2, 2], [8, -1, -0.5, 0], [1, -9, 0, 0]]     
    print("\n\nTest 2: Finding orthonormal basis for arbritrary and more complex vector space {}".format(vectors))     
    ospace = modifiedGramSchmidt(vectors)
    print("\tFor vector space provided \n{}\nOrthonormalized basis produced with modified Gram Schmidt process \n{}".format(vectors, ospace))     
    print("\tAre all basis vectors orthagonal to eachother? {}".format(_is_orthag(ospace))) 
 
if __name__ == "__main__":     
    test() 

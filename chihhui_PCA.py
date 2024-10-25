import numpy as np
from numpy import linalg as LA

class SuperimposePCA2(object):

    def __init__(self, traj_coord):
        self.positions = np.array(traj_coord)
        self.__isSuperimpose = False
        self.__isPCA = False
        
    def superimpose_to_mean(self, cutoff=1e-6, showProcess=True):
        
        if self.__isSuperimpose:
            
            return
        
        round_num = 0
        rmsd = 1
        
        ### translate to (0, 0, 0)
        n_atoms = self.positions.shape[1]
        geometry_center = self.positions.mean(axis=1)
        positions = np.array([pos - np.tile(center, (n_atoms, 1)) for pos, center in zip(self.positions, geometry_center)])

        ### concatonate all the frames into an array, dim0=no. of frames, dim1=no. of atoms, dim2=3(x,y,z coord.)
        mean_struct = np.array([position for position in positions]).mean(axis=0)

        ### when ref. is the first frame
#         mean_struct = positions[0]
        
        while rmsd > cutoff:
            
            rotate_matrix = []
            round_num+=1
            
            ### calculate rotational matrix
            for pos in positions:
                R = mean_struct.T @ pos
                u, diag_s, vh = np.linalg.svd(R)
                
                det = LA.det(u.T @ vh)
                
                m = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, det]])
                
                rotate_matrix.append(u @ m @ vh)
         
            ### get new positions by rotation (r * pos)
            new_positions = np.array([pos @ r.T for r, pos in zip(rotate_matrix, positions)])
            new_mean_struct = new_positions.mean(axis=0)

            ### calculate RMSD
            sd_matrix = (new_mean_struct - mean_struct) ** 2   
            rmsd = (sd_matrix.sum() / np.shape(sd_matrix)[0]) ** 0.5
            
            
            ### change the coordinates and the mean in to coordinates after rotation
            positions = new_positions
            mean_struct = new_mean_struct
        
            if showProcess:
            
                print(str(round_num)+': '+str(rmsd))
        
        self.__isSuperimpose = True
        self.mean_structure = mean_struct
        self.new_positions = new_positions
        
        return

    def eigen_decompose(self):
            
        if self.__isPCA:
                return
    
        if not self.__isSuperimpose:
            self.superimpose_to_mean()

        self.__isPCA = True
        
        ### number of frames(M)
        n_frame = self.positions.shape[0]
        
        ### shape = (M, N, 3)
        shape = self.new_positions.shape
        
        ## size of flat_pos = (M, 3N) M frames, N atoms
        self.flat_pos = np.array(self.new_positions).reshape(shape[0], shape[1]*shape[2])
        q = self.flat_pos
        q_mean = np.tile(q.mean(axis=0), (n_frame, 1))
        
        ### size Q = 3N * M (N atoms, M frames)
        Q = (q - q_mean).T / ((n_frame - 1)**0.5)
        self.Q = Q
        
        ### create covarianve matrix C, size C = 3N * 3N (N atoms)
        C = Q @ Q.T
        
        ### eigen decomposition
        eig_val, eig_vec = LA.eig(C)
            
        ### sort the eigen vals and eigen vecs from large to small
        idx = eig_val.argsort()[::-1]
        sort_eig_val = eig_val[idx]
        sort_eig_vec = eig_vec[:,idx]
        
        ### fill the diagonal elements with sorted eigen values
        eig_vals = np.zeros(C.shape)
#         eig_vals = np.zeros(C.shape,  dtype='complex128')

        np.fill_diagonal(eig_vals, sort_eig_val)
        
        ### discard complex number in eigenvalues/eigenvectors if imaginary parts are samll enough
        eig_vals = np.real_if_close(eig_vals, tol=10**15)
        sort_eig_vec = np.real_if_close(sort_eig_vec, tol=10**15)
        
        ### check if any imaginary parts still remain
        if np.any(np.iscomplex(eig_vals)):
            print("complex number in eig_vals!")
        if np.any(np.iscomplex(sort_eig_vec)):
            print("complex number in sort_eig_vec!")    
        
        self.eigen_vals = eig_vals
        self.eigen_vecs = sort_eig_vec
        
        ### calculat the percentage of variance for pc1 & pc2
        total_var = np.trace(self.eigen_vals)

        ###########
        if np.any(np.iscomplex(total_var)):
            print("complex number in total_var!")
        ##########
        if np.any(np.iscomplex(self.eigen_vals.diagonal()[0])):
            print("complex number in total_var!")
        if np.any(np.iscomplex(self.eigen_vals.diagonal()[1])):
            print("complex number in total_var!")
        
        self.pc1_var = self.eigen_vals.diagonal()[0] / total_var
        self.pc2_var = self.eigen_vals.diagonal()[1] / total_var

        return 


    def project_to_pc(self):

        if not self.__isSuperimpose:
            self.superimpose_to_mean()

        pc1 = self.eigen_vecs[:, 0]
        pc2 = self.eigen_vecs[:, 1]

        ### number of frames(M)
        n_frame = self.positions.shape[0]
        mean = np.tile(self.flat_pos.mean(axis=0), (n_frame, 1))

        ### project to pc1, pc2
        pc1_pos = (self.flat_pos - mean) @ pc1
        pc2_pos = (self.flat_pos - mean) @ pc2
        
        return pc1_pos, pc2_pos
    
    
#########################################################################################################
def superimposition(ref, mobile):

    ### translate center to (0, 0, 0)
    mobile_mean = np.tile(mobile.mean(axis=0), (mobile.shape[0], 1))   ### size = (N, 3)
    mobile_pos = mobile - mobile_mean

    ref_mean = ref
    ref_pos = ref_mean - np.tile(ref_mean.mean(axis=0), (ref_mean.shape[0], 1))    ### size = (N, 3)

    ### calculate rotational matrix
    R = ref_pos.T @ mobile_pos
    u, diag_s, vh = np.linalg.svd(R)

    det = LA.det(u.T @ vh)

    m = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, det]])

    rotate_matrix = u @ m @ vh

    ### get new positions by rotation (r * pos)
    new_mobile_pos = mobile_pos @ rotate_matrix.T

    ### calculate RMSD
    sd_matrix = (new_mobile_pos - ref_pos) ** 2   
    rmsd = (sd_matrix.sum() / np.shape(sd_matrix)[0]) ** 0.5


    return new_mobile_pos, rmsd
        
    
    
    
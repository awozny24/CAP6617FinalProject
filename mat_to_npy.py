# loads matlab .mat file of data, converts to numpy and saves

import numpy as np
import tools

pathSrc = "/home/awozny/SPMat/Source/"
apSource, f0rawSource, n3sgramSource = tools.Load_mat_Files(path, gender=["F"], persInd=[1])

apSource = apSource[0]
f0rawSource = f0rawSource[0]
n3sgramSource = n3sgramSource[0]

apSrcNp = np.concatenate([i.T for i in apSource], axis=0)
f0rawSrcNp = np.concatenate(f0rawSource, axis=0)
n3sgramSrcNp = np.concatenate([i.T for i in n3sgramSource], axis=0)

src = np.concatenate([apSrcNp, f0rawSrcNp, n3sgramSrcNp], axis=1)

np.save("/home/awozny/SPMat/src1.npy", src)

pathTrg = "/home/awozny/SPMat/Target/"
apTarget, f0rawTarget, n3sgramTarget = tools.Load_mat_Files(path, gender=["TrgtF"], persInd=[1])

apTarget = apTarget[0]
f0rawTarget = f0rawTarget[0]
n3sgramTarget = n3sgramTarget[0]

apTrgNp = np.concatenate([i.T for i in apTarget], axis=0)
f0rawTrgNp = np.concatenate(f0rawTarget, axis=0)
n3sgramTrgNp = np.concatenate([i.T for i in n3sgramTarget], axis=0)

trg = np.concatenate([apTrgNp, f0rawTrgNp, n3sgramTrgNp], axis=1)

np.save("/home/awozny/SPMat/trg1.npy", trg)

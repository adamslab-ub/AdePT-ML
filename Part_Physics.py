from jax import vmap, jacfwd, jit,grad
import sys
sys.path.insert(0, '/afs/mae/soumacho/moddiraj/LIPMM_JCISE/')
import Heat_Conduction
import numpy as np
import jax.numpy as jnp
import time

# def Vanilla_FD_JAX_Wrapper(x):
#     # x = jnp.array(x)
#     out = Heat_Conduction.pp_model(x)
#     return out

# def Vanilla_FD_JAX_Wrapper(x):
#     # x = jnp.array(x)
#     out = Heat_Conduction.jacobian(x)
#     return out

def fun_vmap(x,fun):
    x = jnp.array(x); #c = jnp.array(c)
    batched_PP_fwd = vmap(lambda i: fun(x[i,:]))
    out = batched_PP_fwd(jnp.arange(x.shape[0]))
    # out  = jnp.zeros((x.shape[0],2))
    # for i in range(x.shape[0]):
    #     out = out.at[i,:].set(Heat_Conduction.pp_model(x[i,:]))
    return np.array(out)

def grad_fun_vmap(x,fun):
    x = jnp.array(x); #c =jnp.array(c)
    t1 = time.time()
    jaco = lambda i: fun(x[i,:])
    batched_jacobian = vmap(jaco)
    out = batched_jacobian(jnp.arange(x.shape[0]))
    t2 = time.time()
    # print(t2 - t1, "jax grad computation time")
    return np.array(out)



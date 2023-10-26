from jax import vmap, jacfwd, jit,grad
import sys
sys.path.insert(0, '/afs/mae/soumacho/moddiraj/LIPMM_JCISE/')
import numpy as np
import jax
import jax.numpy as jnp
import time

def physics_fwd(x):
    n = 3
    an = lambda x,n: jnp.sum(jnp.array([(4/(i *jnp.pi)) * jnp.sin(i*x*jnp.pi) for i in range(1,n+1,2)]))
    a = 1
    return a + an(x,n)

grad = jax.grad(physics_fwd)

def vmap_wrapper(x,fun):
    x = jnp.array(x); #c = jnp.array(c)
    batched_fwd = vmap(lambda x_t: fun(x_t))
    out = batched_fwd(x)
    return np.array(out).reshape(-1,1)

batch_physics = lambda x: vmap_wrapper(x,physics_fwd)
batch_jacobian = lambda x: vmap_wrapper(x,grad)

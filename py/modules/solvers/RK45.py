import numpy as np
from .solver_utilities import OdeResult
# =====  Coefficients for RK45 Initialization =====
# c: Coefficients for time steps advancement
c = [0, 1/4, 3/8, 12/13, 1, 1/2]
# a: Coefficients for calculating k values based on previous k values
a = [[],
    [1/4],
    [3/32, 9/32],
    [1932/2197, -7200/2197, 7296/2197],
    [439/216, -8, 3680/513, -845/4104],
    [-8/27, 2, -3544/2565, 1859/4104, -11/40]]    
# b: 4th order solution coefficients
b = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
# b_star: 5th order solution coefficients
b_star = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]    

def rk45(fun,
         t_span,
         Y0,
         *,
         args=(),
         events=None,
         h0=0.1,
         h_min=1e-6,
         h_max=1.0,
         rtol=1e-6,
         atol=1e-9,
         h_adaptative=True,
         max_iter=100000,
         verbose=False):
    
    # ===== Initialize variables =====
    # Time setup
    t0, tf = t_span
    # Initial setup
    t = t0
    h = h0    
    Y = np.array(Y0, dtype=float)
    # Results setup
    ts = [t0]
    Ys = [Y.copy()]
    nfev = 0
    message = ""
    status = 0    
    # Error control parameters
    safety = 0.9
    p = 5.0   # order
    
    # ===== Event handling setup =====
    terminated_by_event = False

    if events is not None:
        if callable(events):
            events = [events]

        g_prev = [ev(t, Y, *args) for ev in events]

        t_prev = t
        Y_prev = Y.copy()

        t_events = [[] for _ in events]
        y_events = [[] for _ in events]

    else:
        g_prev = None
        t_events = []
        y_events = []
    def handle_events(t, Y, t_prev, Y_prev):
        if events is None:
            return False, None

        for i, ev in enumerate(events):

            g_new = ev(t, Y, *args)

            crossed = (
                (g_prev[i] <= 0 and g_new >= 0) or
                (g_prev[i] >= 0 and g_new <= 0)
            )

            direction = getattr(ev, "direction", 0)

            if direction == 1:
                crossed &= g_new > g_prev[i]
            elif direction == -1:
                crossed &= g_new < g_prev[i]

            if crossed:

                alpha = abs(g_prev[i]) / (abs(g_prev[i]) + abs(g_new))

                t_event = t_prev + alpha*(t - t_prev)
                y_event = Y_prev + alpha*(Y - Y_prev)

                t_events[i].append(t_event)
                y_events[i].append(y_event.copy())

                if getattr(ev, "terminal", False):

                    return True, i

            g_prev[i] = g_new

        return False, None

    # ===== Main integration loop =====
    iter_count = 0
    while t < tf and iter_count < max_iter:
        if t + h > tf: # Adjust final step size to not exceed tf
            h = tf - t

        # ----- Store previous step values for event handling -----
        t_prev = t
        Y_prev = Y.copy()

        # ----- Calculate k values -----
        k = np.zeros((6, len(Y))) # k1 to k6
        k[0] = fun(t, Y, *args)
        nfev += 1
        for i in range(1, 6):
            Yi = Y.copy()
            for j in range(i):
                Yi += h * a[i][j] * k[j]

            k[i] = fun(t + c[i]*h, Yi, *args)
            nfev += 1

        # ----- Calculate 4th and 5th order solutions -----
        Y4 = Y + h * np.dot(b, k)
        Y5 = Y + h * np.dot(b_star, k)

        if h_adaptative: # Adaptive step size control
            # ----- Estimate error for adaptive step size -----
            scale = atol + rtol * np.maximum(np.abs(Y), np.abs(Y5))
            err = np.sqrt(
                np.mean(((Y5 - Y4) / scale)**2)
            )

            # ----- Step control based on error -----
            if err <= 1.0:
                # Accept
                t += h
                Y = Y5

                ts.append(t)
                Ys.append(Y.copy())
                if verbose:
                    print("----- Step Accepted -----")
                    print("Iteration %d: t = %.6f, h = %.6f, err = %.6e" % (iter_count, t, h, err))
                    print(Y)
                    print("-------------------------")

                # New step
                if err == 0:
                    factor = 2.0
                else:
                    factor = safety * err**(-1/p)

                h *= np.clip(factor, 0.2, 5.0)

                terminate, ev_id = handle_events(t, Y, t_prev, Y_prev)
                if terminate:
                    status = 1
                    message = f"Terminated by event {ev_id}"
                    terminated_by_event = True
                    success = True
                    break
                
            else:
                # Reject
                factor = safety * err**(-1/p)
                h *= np.clip(factor, 0.1, 0.5)

            h = np.clip(h, h_min, h_max)

            if h <= h_min and err > 1.0:
                raise RuntimeError("Step size underflow: integration failed")

        else: # Fixed step size
            # ----- Always accept step -----            
            t += h
            Y = Y5
            ts.append(t)
            Ys.append(Y.copy())
            h = h0

            terminate, ev_id = handle_events(t, Y, t_prev, Y_prev)
            if terminate:
                status = 1
                message = f"Terminated by event {ev_id}"
                success = True
                terminated_by_event = True
                break

        # Next iteration
        iter_count += 1
    
    # ===== Format results with a similar structure to SciPy's ODE solvers for compability =====
    if terminated_by_event:
        pass # already handled in the loop                        

    elif iter_count >= max_iter and t < tf:
        success = False
        status = -1
        message = "Maximum number of iterations exceeded."

    else:
        success = True
        status = 0
        message = "Integration successful."
    

    # ===== Return results =====
    result = OdeResult()
    result.t = np.array(ts)
    result.y = np.array(Ys).T
    result.success = success
    result.status = status
    result.message = message
    result.nfev = nfev
    result.njev = 0 # Not implemented
    result.nlu = 0 # Not implemented
    result.t_events = [np.array(te) for te in t_events]
    result.y_events = [np.array(ye) for ye in y_events]

    return result
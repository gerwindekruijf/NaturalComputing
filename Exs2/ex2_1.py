import numpy as np

def func(x):
    return np.sum([(-1*x[0]) * np.sin(np.sqrt(np.abs(x[0]))), (-1*x[1]) * np.sin(np.sqrt(np.abs(x[1])))])

# print("(-400, -400): ", func([-400,-400]))
# print("(-410, -410): ", func([-410,-410]))
# print("(-415, -415): ", func([-415,-415]))

#to update the first time for the exercise, noting that individual best are the individuals and alphas, r's are given
def update_rule_once(x, v, omega, best):
    # r1 = r2 = r
    r = 0.5
    #a1 = a2 = a
    alpha = 1
    new_v = omega * v + alpha * r * (best - x) 
    x = x + new_v
    for i in range(2):
        if x[i] > 500:
            x[i] = 500
        elif x[i] < -500:
            x[i] = -500
    return x 

#initial positions
x1 = np.array([-400,-400])
x2 = np.array([-410,-410])
x3 = np.array([-415,-415])

#velocities
v1 = np.array([-50,-50])
v2 = np.array([-50,-50])
v3 = np.array([-50,-50])

for om in [2,0.5,0.1]:
    print("Value of omega: ", om)
    x1_up = update_rule_once(x1,v1, om, x3)
    print("x1 After first update: ", x1_up, "fitness: ", func(x1_up))
    x2_up = update_rule_once(x2,v1, om, x3)
    print("x2 After first update: ", x2_up, "fitness: ", func(x2_up))
    x3_up = update_rule_once(x3,v1, om, x3)
    print("x3 After first update: ", x3_up, "fitness: ", func(x3_up))



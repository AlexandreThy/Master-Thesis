# Master Thesis Code Usage

Each notebook represents a different experiment made with my model. 

AdaptiveFBLin runs my model with a nonlinear adaptive feedback controller : 
Use the fonction run(Number of trials for the adaptation,Basis function of the adaptive controller,Type of error function for the controller,Learning rate,factor = OPTIMAL_FACTORS[11])
We can also allow here to use a nonlinear model for torque dynamics with the parameters NLTorque = True


Experiment_Comité runs my model that performs the square-rectangle task made by Antoine de comité :
Use the function Simulation_Mec(perturbation torque (1x2 array),weight for shoulder angle,weight for elbow angle,weight for shoulder angular velocity,weight for elbow angular velocity,additional weight (do not use),weight on command for shoulder angle,weight on command for elbow angle,Type of trial)
for the type of trial, use s for square and r for rectangle such that "sr" is a square that transforms in rectangle. "ss" is a square that stays square,...

Experiment Multitarget runs my model on multitarget reaching, where the person had to chose between two targets, with perturbation and change in rewards on each target.
Use the function Simulation_Multitarget(weight for shoulder angle,weight for elbow angle,weight for shoulder angular velocity,weight for elbow angular velocity,targets angle coordinates,Initial reward on each target,additional reward on each  target,Boolean allowing to adapt the target choice during movement,Number of targets)

ForceField was used to match my model with ForceField data
Use the function Simulation_FF(weight for shoulder angle,weight for elbow angle,weight for shoulder angular velocity,weight for elbow angular velocity,Learning state for this simulation within the range [0,1],starting coordinates in X;Y ,targets = target coordinate in X;Y,proportionnality=factor,Boolean for allowing to plot ,Y value where the perturbation starts)

Gravity was used to perform reaching movements with internal model of gravity that does not match reality.
Use Simulation_Gravity(weight for shoulder angle,weight for elbow angle,weight for shoulder angular velocity,weight for elbow angular velocity,starting coordinates in X;Y ,targets = target coordinate in X;Y,color of the lineplot,label of the lineplot , body angle compared to gravity)

Velocity profile was just used to check specific velocity profiles on some trajectories, to be compared with the paper on MinimumJerk


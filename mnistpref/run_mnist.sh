project=symloss_report_for_paper

#pn
# instance independent symmetric noise
add_noise=True
symmetric=True
instance_dependent_noise=False
flipping=False
class_prior=0.5
for seed in 0 1 2 3 4; do
    for epsilon in 0.1 0.2 0.3 0.4; do
        for clip in 20; do
            for algo in pn; do
                for loss_type in logistic hinge square sigmoid symmetric_ramp unhinged ramp; do
                    python train.py seed=$seed algo=$algo loss_type=$loss_type project=$project regularize=True symmetric=$symmetric instance_dependent_noise=$instance_dependent_noise epsilon_p=$epsilon epsilon_n=$epsilon add_noise=$add_noise flipping=$flipping class_prior=$class_prior clip_min=-$clip clip_max=$clip 
                done
            done
        done
    done
done

# sympo
add_noise=True
symmetric=True
instance_dependent_noise=False
flipping=False
class_prior=0.5
for seed in 0 1 2 3 4; do
    for epsilon in 0.1 0.2 0.3 0.4; do
        for clip in 20; do
            for algo in sympo; do
                for loss_type in logistic hinge square sigmoid symmetric_ramp unhinged ramp; do
                    python train.py seed=$seed algo=$algo loss_type=$loss_type project=$project regularize=True symmetric=$symmetric instance_dependent_noise=$instance_dependent_noise epsilon_p=$epsilon epsilon_n=$epsilon add_noise=$add_noise flipping=$flipping class_prior=$class_prior clip_min=-$clip clip_max=$clip 
                done
            done
        done
    done
done
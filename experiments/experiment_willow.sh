for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_sim.launch script:=nav_cloning_with_direction_intersection_node_branch_fast.py mode:=selected_training
  sleep 10
done

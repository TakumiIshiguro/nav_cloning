for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_sim.launch script:=nav_cloning_with_direction_intersection_node_branch_on.py mode:=selected_training world_name:=tsudanuma2-3_v2.3.3.world map_file:=real_tsudanuma2-3_v2.yaml waypoints_file:=cit3f_rotation.yaml dist_err:=1.0 initial_pose_x:=0.0 initial_pose_y:=0.0 initial_pose_a:=1.57 use_waypoint_nav:=true robot_x:=-5.0 robot_y:=7.7 robot_Y:=-1.57
  sleep 10
done

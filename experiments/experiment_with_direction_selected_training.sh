for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_sim.launch script:=nav_cloning_with_direction_intersection_node_branch_on.py mode:=selected_training world_name:=cross_road.world map_file:=cross_road.yaml waypoints_file:=cross_road_way_cmd.yaml dist_err:=1.0 initial_pose_x:=0 initial_pose_y:=0 use_waypoint_nav:=true use_initpose:=true robot_x:=-5.0 robot_y:=7.7 robot_Y:=3.14
  sleep 10
done

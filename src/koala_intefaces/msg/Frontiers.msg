std_msgs/Header header

# Individual frontier points
Point2DKoala[] frontiers

# Cluster centroids or representative points
Point2DKoala[] clusters

# Map metadata
float64 origin_x
float64 origin_y
float64 resolution

# Performance metrics
float64 cpu_time
float64 cpu_usage
float64 memory_usage
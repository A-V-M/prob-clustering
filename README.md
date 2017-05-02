# prob-clustering
Unsupervised clustering based on Lloyd’s algorithm. 
Unsupervised clustering algorithm based on Lloyd’s algorithm. 
Key points:
•	works only with binary vectors, 
•	example-centroid distances are estimated as the norm of differences between a feature and its likelihood, p(x|y)
•	performance is assessed as the mean entropy across all examples

To-do list:

1.	Optimise code to exclude all for loops, 
2.	find a better way to deal with datasets with linearly dependent columns. At present it will simply compute the determinant of the co-variance matrix between all features and issue a warning if it's non-positive.

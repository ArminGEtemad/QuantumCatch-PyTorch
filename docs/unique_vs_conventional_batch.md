# Benchmarking
In this document, I benchmark the two ways I calculate the propagator. Once is the conventional batches and vectorizing the matrix multiplications. The other way is "**unique_batches**" where we avoid the duplicates. From now on, I will call them `batch` and `unique` method. It is important to keep in mind that inside the `unique` method there is still the `batch` method.

ARM64 MacBook : unique
|Resolution |   Time (s)|
|-----------|------------|
 |      128 |   0.060148|
 |      256 |   0.205889|
 |      512 |   0.783300|
 |     1024 |   3.167536|
 |     2048 |  17.085888|
 |     4096 | 104.258061|

ARM64 MacBook : batch
| Resolution |   Time (s)|
|-------------|-----------|
|       128 |   0.071030|
|       256 |   0.245407|
|       512 |   0.917591|
|      1024 |   3.574566|
|      2048 |  15.289475|
|      4096 | 115.277794|

clearly something is happening between 2K and 4K. So I decided to run the tests tree times and then calculate the average

ARM64 MacBook : unique
|Resolution |   Avg Time (s)|
|-------------|-----------|
|        32 |       0.013199|
|        64 |       0.027243|
 |      128 |       0.063169|
|       256 |       0.260320|
 |      512 |       0.776440|
 |     1024 |       3.104264|
 |     2048 |      13.534600|
 |     4096 |      96.594893|


ARM64 MacBook : batch
|Resolution |   Avg Time (s)|
|-------------|-----------|
|        32 |       0.016276|
|        64 |       0.030568|
|       128 |       0.070643|
|       256 |       0.240050|
|       512 |       0.941397|
|      1024 |       3.480321|
|      2048 |      15.100044|
|      4096 |     110.206708|
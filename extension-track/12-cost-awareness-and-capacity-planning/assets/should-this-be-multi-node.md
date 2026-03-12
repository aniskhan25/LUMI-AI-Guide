# Should This Be Multi-Node? Worksheet

Answer `yes/no` to each:

- does single-node baseline miss target throughput materially?
- is bottleneck confirmed as compute/communication and not I/O?
- is effective workload normalized for fair comparison?
- are preprocessing/model artifacts reusable across trials?
- is there a clear stop condition if gains are small?

If two or more answers are `no`, hold at single-node and fix bottlenecks first.

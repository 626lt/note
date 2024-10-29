# Deadlock

## Deadlock problem

Deadlock problem: a set of block processes each holding a resource and waiting to acquire a resource held by another process in the set.

Note: most OSes do not prevent or deal with deadlocks.

Can be illustrated by a resource allocation graph.

## System Model of deadlock

+ Resources: each represents a different resource type e.g. CPU cycles, memory space, I/O devices
+ each resource type $R_i$ has $W_i$ instances
+ Each process utilizes a resource in the following pattern
  + request
  + use
  + release

### Four Conditions of Deadlock

1. Mutual exclusion: a resource can only be used by one process at a time.
2. Hold and wait: a process holding at least one resource is waiting to acquire additional resources held by other processes.
3. No preemption: a resource can be released only voluntarily by the process holding it, after it has completed its task.
4. Circular wait: there exists a set of waiting processes 


## How to handle deadlock

Ensure the system will never enter a deadlock state.

+ Prevention
+ Avoidance

Allow the system to enter a deadlock state and then recover.

+ Detection

Ignore the problem and pretend that deadlocks never occur in the system; used by most OSes.


### Deadlock Prevention

+ prevent mutual exclusion
  + not required for sharable resources
  + must hold for non-sharable resources
+ hold and wait
  + whenever a process requests a resource, it doesn’t hold any other resources
    + require process to request all its resources before it begins execution
    + allow process to request resources only when the process has none
      申请资源时不能有其他资源，要一次性申请所有需要的资源。
  + low resource utilization; starvation possible
    利用率低，而且可能有进程永远拿不到所有需要的资源，因此无法执行。
+ no preemption
  + 可以抢，但不实用。
+ circular wait
  + impose a total ordering of all resource types
  + require that each process requests resources in an increasing order of enumeration
  + many OSes adopt this strategy for some locks.

### Deadlock Avoidance

avoidance 用了一些算法，在分配资源之前，先判断是否会死锁，如果会死锁就不分配。

#### Safe State

序列里的每一个进程都可以被满足。（空闲的资源和之前的进程释放的资源）
Safe state can guarantee no deadlock.
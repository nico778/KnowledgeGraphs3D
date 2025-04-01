//check how cuda works underneath python abstraction

#pragma omp for [clause ...]  //implicit barrier
    // Loop iterations are divided into pieces of size chunk and
    dynamically scheduled among the threads; when a thread
    finishes one chunk, it is dynamically assigned another
    schedule (type [,chunk])
    ordered
    private (list)
    firstprivate (list)
    lastprivate (list)
    shared (list)
    reduction (operator: list)
    collapse (n)
    // Threads do not synchronize at the end of the parallel loop.
    nowait

<for_loop>


#pragma omp sections [clause ...]
    private (list)
    firstprivate (list)
    lastprivate (list)
    reduction (operator: list)
    nowait

{
 #pragma omp section
 <structured_block>
 #pragma omp section
 <structured_block>
 ...
}

// check how this works in pytorch or if beneficial for image processing!
int histo[MAX_THREADS];
#pragma omp parallel private(i)
{
    int nthreads, myid;
    nthreads = omp_get_num_threads();
    myid = omp_get_thread_num();
    for( i=0; i<N; i++ ) {
        if( A[i]%nthreads == myid ) {
            histo[myid]++;
        }
    }
}


int histo[MAX_THREADS];
#pragma omp parallel private(i)
{
    int nthreads, myid;
    int count=0;
    nthreads = omp_get_num_threads();
    myid = omp_get_thread_num();
    for( i=0; i<N; i++ ) {
        if( A[i]%nthreads == myid ) {
            count++;
        }
    }
    histo[myid]=count;
}


struct mycount
{int count; char padding[60];};
struct mycount padhisto[MAXTH];
#pragma omp parallel private(i)
{
    int nthreads, myid;
    nthreads = omp_get_num_threads();
    myid = omp_get_thread_num();
    for( i=0; i<N; i++ ) {
        if( A[i]%nthreads == myid ) {
            padhisto[myid].count++; }
    }
}   

//Tasks how fine grained can we even go in pytorch
default (shared|none)
private (list)
firstprivate (list)
shared (list)
if (logical-expr.)
final (logical-expr.)
mergeable
depend (dependence-type:list)
priority (priority-value)
untied

//mpi program structure, compatible with pytorch?
MPI_INIT
– MPI_FINALIZE
– MPI_COMM_SIZE
– MPI_COMM_RANK
– MPI_SEND
– MPI_RECV

RMA in MPI? maybe depends how many gpus are available
● One sided
● Remote Memory Access
● MPI_Put and MPI_Get
● only one process (called the “origin”) actively participates in the data transfer
● Guarantees?

Active / Passive Target Synchronization:
● Fence
○ MPI_Win_fence()
○ Collective call over all processes in the group associated with the window object
● PSCW
○ MPI_Win_post()
○ MPI_Win_start()
○ MPI_Win_complete()
○ MPI_Win_wait()
○ More general
○ Can select between exposure and access epochs
● Passive-Target
○ MPI_Win_lock()
○ MPI_Win_unlock()
○ Target does not call any sync routines (hence passive)


MPI Parameters (Put, Get, Send/Rec)
● Put/Get:
○ origin_addr - initial address of origin buffer (choice) || Address of the buffer in
which to receive the data
○ origin_count - number of entries in origin buffer (nonnegative integer)
○ origin_datatype datatype of each entry in origin buffer (handle)
○ Target_rank rank of target (nonnegative integer)
○ Target_disp displacement from start of window to target buffer (nonnegative
integer) || displacement from window start to the beginning of the target buffer
(nonnegative integer)
○ Target_count number of entries in target buffer (nonnegative integer)
○ Target_datatype datatype of each entry in target buffer (handle)
○ Win window object used for communication (handle)
● Send
○ Buf initial address of send buffer (choice)
○ Count number of elements in send buffer (nonnegative integer)
○ Datatype datatype of each send buffer element (handle)
○ Dest rank of destination (integer)
○ Tag message tag (integer)
○ Comm communicator (handle)
● Recv
○ Count maximum number of elements in receive buffer (integer)
○ Datatype datatype of each receive buffer element (handle)
○ Source rank of source (integer)
○ Tag message tag (integer)
○ Comm communicator (handle)

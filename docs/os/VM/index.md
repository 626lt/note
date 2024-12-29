# VM

ELF Entry: _start

partially-loaded program

## Demand Paging

+ Demand Paging vs page fault
  + Demand Paging: load page when needed, not malloc all memory at the beginning 
+ What causes page fault?
  + User space program accesses an address
+ Which hardware issues a page fault?
  + MMU
+ Who handles page fault?
  + OS
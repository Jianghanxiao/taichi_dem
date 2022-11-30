from TypeDefine import *
import time

class DEMSolverStatistics:
    class Timer:
        def __init__(self):
            self.first:bool = True
            self.on:bool = False
            self.start:float = 0.0
            self.total = 0.0

        def tick(self):
            ti.sync()
            if(self.on == False): 
                self.start = time.time()
                self.on = True
            else:
                if(self.first): self.first = False
                else: self.total += time.time() - self.start
                self.on = False
        
        def __str__(self):
            return str(self.total)
        
    def __init__(self):
        self.SolveTime = self.Timer()
        
        self.BroadPhaseDetectionTime = self.Timer()
        self.HashTableSetupTime = self.Timer()
        self.PrefixSumTime = self.Timer()
        self.CollisionPairSetupTime = self.Timer()
        
        self.ContactResolveTime = self.Timer()
        self.ContactTime = self.Timer()
        self.ResolveWallTime = self.Timer()
        self.ApplyForceTime = self.Timer()
        self.UpdateTime = self.Timer()
        
    
    def _pct(self, x:Timer):
        if(self.SolveTime.total == 0.0): return '0%'
        return str(x.total / self.SolveTime.total * 100) + '%'
    
    def report(self):
        print(f"Total              = {self.SolveTime}\n"
              f"ApplyForceTime     = {self.ApplyForceTime}({self._pct(self.ApplyForceTime)})\n"
              f"UpdateTime         = {self.UpdateTime}({self._pct(self.UpdateTime)})\n"
              f"ResolveWallTime    = {self.ResolveWallTime}({self._pct(self.ResolveWallTime)})\n"
              f"ContactTime        = {self.ContactTime}({self._pct(self.ContactTime)})\n"
              f"    -BPCD               = {self.BroadPhaseDetectionTime}({self._pct(self.BroadPhaseDetectionTime)})\n"
              f"        --HashTableSetupTime      = {self.HashTableSetupTime}({self._pct(self.HashTableSetupTime)})\n"
              f"        --PrefixSumTime           = {self.PrefixSumTime}({self._pct(self.PrefixSumTime)})\n"
              f"        --CollisionPairSetupTime  = {self.CollisionPairSetupTime}({self._pct(self.CollisionPairSetupTime)})\n"
              f"    -ContactResolveTime = {self.ContactResolveTime}({self._pct(self.ContactResolveTime)})\n"
              )

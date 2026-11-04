from dataclasses import dataclass
from enum        import auto, Enum
from random      import Random
from netdef      import Grid

ansiblu = "\033[..."
ansired = "\033[..."
ansirst = "\033[..."
ansicls = "\033[..."

randgen = Random(42)


class Move(Enum):
    Up = auto()
    Dn = auto()
    Lf = auto()
    Rg = auto()


@dataclass
class Spec:
    rownum: int = 42
    rowini: int = 37

    colnum: int = 37
    colini: int = 17

    radius: int = 1

    bluini: int = 42
    bludif: int = 1
    bluval: int = 0
    blustr: str = ansiblu + "B" + ansirst

    redini: int = 0
    reddif: int = 1
    redval: int = 1
    redstr: str = ansired + "R" + ansirst

    youval: int = 2
    youstr: str = ansirst + "Y" + ansirst


def tilecol(spec: Spec, probblu: float, probred: float) -> int:
    return randgen.choices(
        [spec.bluval, spec.redval],
        [probblu    , probred    ],
    )[0]


def numgrid(spec: Spec) -> Grid[int]:
    probblu = spec.bluini
    probred = spec.redini
    grid    = []

    for _ in range(spec.rownum):
        probblu -= 0 if probblu <= 0 else spec.bludif
        probred += 0 if probred >  0 else spec.reddif
        temp = [tilecol(spec, probblu, probred) for _ in range(spec.colnum)]
        grid.append(temp)
    return grid


def strgrid(spec: Spec, grid: Grid[int]) -> str:
    youblu = spec.youval + spec.bluval
    youred = spec.youval + spec.redval
    repr = ""

    for row in grid:
        temp  = "\n"
        repr += temp

        for cell in row:
            if cell == youblu:
                temp += spec.youstr
            if cell == youred:
                temp += spec.youstr
            if cell == spec.bluval:
                temp += spec.blustr
            if cell == spec.redval:
                temp += spec.redstr

    return repr


class Env:
    def __init__(self, spec: Spec = Spec()) -> None:
        self.rownow = spec.rowini
        self.colnow = spec.colini
        self.spec   = spec
        self.grid   = numgrid(spec)

    def __str__(self) -> str:
        return strgrid(
            self.spec, self.view()
        )

    def done(self) -> bool:
        return bool(
            self.rownow == self.spec.rownum - 1
        )

    def move(self, move: Move) -> None:
        if move == Move.Up:
            self.rownow-= 1
        if move == Move.Dn:
            self.rownow+= 1
        if move == Move.Lf:
            self.colnow -= 1
        if move == Move.Rg:
            self.colnow += 1

        self.rownow = max(0, min(self.rownow, self.spec.rownum - 1))
        self.colnow = max(0, min(self.colnow, self.spec.colnum - 1))

    def view(self) -> Grid[int]:
        grid = [row[:] for row in self.grid]
        grid[self.rownow]\
            [self.colnow]\
        += self.spec.youval
        return grid

    def near(self) -> Grid[int]:
        grid = self.view()

        rowmax = min(self.spec.rownum, self.rownow + self.spec.radius + 1)
        rowmin = max(0, self.rownow - self.spec.radius)
        colmax = min(self.spec.colnum, self.colnow + self.spec.radius + 1)
        colmin = max(0, self.colnow - self.spec.radius)

        return [row[colmin:colmax] for row in grid[rowmin:rowmax]]
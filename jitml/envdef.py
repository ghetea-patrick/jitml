from dataclasses import dataclass
from enum import auto, Enum
from random import Random

from netdef import Grid

ansiblu = "\033[34m"
ansired = "\033[31m"
ansirst = "\033[0m"
ansicls = "\033[2J\033[H"

randgen = Random(42)


class Move(Enum):
    Up = auto()
    Dn = auto()
    Lf = auto()
    Rg = auto()


@dataclass
class Specs:
    bluini: int = 16
    bludif: int = 1
    bluval: int = 0
    blustr: str = ansiblu + "B" + ansirst

    redini: int = 0
    reddif: int = 1
    redval: int = 1
    redstr: str = ansired + "R" + ansirst

    youval: int = 2
    youstr: str = "Y"

    rownum: int = 16
    rowini: int = 8

    colnum: int = 16
    colini: int = 0

    sedval: int = 42
    radius: int = 1


def numgrid(specs: Specs) -> Grid[int]:
    probblu = specs.bluini
    probred = specs.redini
    newgrid = []
    tilpick = 0

    for _ in range(specs.rownum):
        probblu -= specs.bludif if probblu > 0 else 0
        probred += specs.reddif if probred >= 0 else 0

        temprow = []
        for _ in range(specs.colnum):
            tilpick = randgen.choices([specs.bluval, specs.redval], [probblu, probred])[0]
            temprow.append(tilpick)
        newgrid.append(temprow)

    return newgrid


def strgrid(grid: Grid[int], specs: Specs) -> str:
    newgrid = []

    blu_youval = specs.youval + specs.bluval
    red_youval = specs.youval + specs.redval

    for row in grid:
        temprow = ""
        for cel in row:
            if cel == specs.bluval:
                temprow += specs.blustr
            if cel == specs.redval:
                temprow += specs.redstr
            if cel == blu_youval:
                temprow += specs.youstr
            if cel == red_youval:
                temprow += specs.youstr
        newgrid.append(temprow)

    return "\n".join(newgrid)


class Env:
    def __init__(self, specs: Specs = Specs()) -> None:
        self.mygrid = numgrid(specs)
        self.myspec = specs
        self.rowcur = specs.rowini
        self.colcur = specs.colini

    def __str__(self) -> str:
        return strgrid(self.view(), self.myspec)

    def done(self) -> bool:
        return bool(self.rowcur == self.myspec.rownum - 1)

    def move(self, move: Move) -> None:
        if move == Move.Up:
            self.rowcur -= 1
        if move == Move.Dn:
            self.rowcur += 1
        if move == Move.Lf:
            self.colcur -= 1
        if move == Move.Rg:
            self.colcur += 1

        self.rowcur = max(0, min(self.rowcur, self.myspec.rownum - 1))
        self.colcur = max(0, min(self.colcur, self.myspec.colnum - 1))

    def view(self) -> Grid[int]:
        grid = [row[:] for row in self.mygrid]
        grid[self.rowcur][self.colcur] += self.myspec.youval
        return grid

    def near(self) -> Grid[int]:
        grid = self.view()

        rowmax = min(self.myspec.rownum, self.rowcur + self.myspec.radius + 1)
        rowmin = max(0, self.rowcur - self.myspec.radius)

        colmax = min(self.myspec.colnum, self.colcur + self.myspec.radius + 1)
        colmin = max(0, self.colcur - self.myspec.radius)

        return [row[colmin:colmax] for row in grid[rowmin:rowmax]]
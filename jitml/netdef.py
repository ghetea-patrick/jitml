from torch import Tensor

import defval
from envdef import Move

type Mnn = Snn | Ann | Wnn | Bnn
type Que = list[Tnn]

type Grid[T] = list[list[T]]
type Cube[T] = list[Grid[T]]
type Tess[T] = list[Cube[T]]

type Ten = Tensor


class Net:
    def __init__(
        self,
        layersize: list[int],
        activfunc: list,
        weigvalue: Cube[float],
        biasvalue: Grid[float],
        currinput: Grid[int],
        pastinput: Cube[int],
        lifescore: int,
        lifelimit: int,
        goodmatch: bool
    ) -> None:
        self.layersize = layersize
        self.activfunc = activfunc
        self.weigvalue = weigvalue
        self.biasvalue = biasvalue
        self.currinput = currinput
        self.pastinput = pastinput
        self.lifescore = lifescore
        self.lifelimit = lifelimit
        self.goodmatch = goodmatch

    def forward(self, ten: Ten) -> ...:
        ...

    def trainme(self, que: Que) -> ...:
        ...


class Tnn(Net):
    def __init__(
        self,
        layersize: None | list[int]   = None,
        activfunc: None | list        = None,
        weigvalue: None | Cube[float] = None,
        biasvalue: None | Grid[float] = None,
        currinput: None | Grid[int]   = None,
        pastinput: None | Cube[int]   = None,
        lifescore: None | int         = None,
        lifelimit: None | int         = None,
        goodmatch: None | bool        = None
    ) -> None:
        super().__init__(
            layersize or defval.TNN_LAYERSIZE,
            activfunc or defval.TNN_ACTIVFUNC,
            weigvalue or defval.TNN_WEIGVALUE,
            biasvalue or defval.TNN_BIASVALUE,
            currinput or [],
            pastinput or [],
            lifescore or 10,
            lifelimit or 0,
            goodmatch or False)

    def gen_takenmove(self) -> Move:
        ...


class Snn(Net):
    def __init__(
        self,
        layersize: None | list[int]   = None,
        activfunc: None | list        = None,
        weigvalue: None | Cube[float] = None,
        biasvalue: None | Grid[float] = None,
        currinput: None | Grid[int]   = None,
        pastinput: None | Cube[int]   = None,
        lifescore: None | int         = None,
        lifelimit: None | int         = None,
        goodmatch: None | bool        = None
    ) -> None:
        super().__init__(
            layersize or defval.SNN_LAYERSIZE,
            activfunc or defval.SNN_ACTIVFUNC,
            weigvalue or defval.SNN_WEIGVALUE,
            biasvalue or defval.SNN_BIASVALUE,
            currinput or [],
            pastinput or [],
            lifescore or 10,
            lifelimit or 0,
            goodmatch or False)

    def gen_layersize(self) -> list[int]:
        ...


class Ann(Net):
    def __init__(
        self,
        layersize: None | list[int]   = None,
        activfunc: None | list        = None,
        weigvalue: None | Cube[float] = None,
        biasvalue: None | Grid[float] = None,
        currinput: None | Grid[int]   = None,
        pastinput: None | Cube[int]   = None,
        lifescore: None | int         = None,
        lifelimit: None | int         = None,
        goodmatch: None | bool        = None
    ) -> None:
        super().__init__(
            layersize or defval.ANN_LAYERSIZE,
            activfunc or defval.ANN_ACTIVFUNC,
            weigvalue or defval.ANN_WEIGVALUE,
            biasvalue or defval.ANN_BIASVALUE,
            currinput or [],
            pastinput or [],
            lifescore or 10,
            lifelimit or 0,
            goodmatch or False)

    def gen_activfunc(self) -> list:
        ...

class Wnn(Net):
    def __init__(
        self,
        layersize: None | list[int]   = None,
        activfunc: None | list        = None,
        weigvalue: None | Cube[float] = None,
        biasvalue: None | Grid[float] = None,
        currinput: None | Grid[int]   = None,
        pastinput: None | Cube[int]   = None,
        lifescore: None | int         = None,
        lifelimit: None | int         = None,
        goodmatch: None | bool        = None
    ) -> None:
        super().__init__(
            layersize or defval.WNN_LAYERSIZE,
            activfunc or defval.WNN_ACTIVFUNC,
            weigvalue or defval.WNN_WEIGVALUE,
            biasvalue or defval.WNN_BIASVALUE,
            currinput or [],
            pastinput or [],
            lifescore or 10,
            lifelimit or 0,
            goodmatch or False)

    def gen_weigvalue(self) -> Cube[float]:
        ...

class Bnn(Net):
    def __init__(
        self,
        layersize: None | list[int]   = None,
        activfunc: None | list        = None,
        weigvalue: None | Cube[float] = None,
        biasvalue: None | Grid[float] = None,
        currinput: None | Grid[int]   = None,
        pastinput: None | Cube[int]   = None,
        lifescore: None | int         = None,
        lifelimit: None | int         = None,
        goodmatch: None | bool        = None
    ) -> None:
        super().__init__(
            layersize or defval.BNN_LAYERSIZE,
            activfunc or defval.BNN_ACTIVFUNC,
            weigvalue or defval.BNN_WEIGVALUE,
            biasvalue or defval.BNN_BIASVALUE,
            currinput or [],
            pastinput or [],
            lifescore or 10,
            lifelimit or 0,
            goodmatch or False)

    def gen_biasvalue(self) -> Grid[float]:
        ...
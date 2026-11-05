from   torch  import Tensor
from   envdef import Move
from   tgpdef import Elec
import defval

type Mnn = Snn | Ann | Wnn | Bnn
type Que = list[Tnn]

type Grid[T] = list[list[T]]
type Cube[T] = list[Grid[T]]
type Tess[T] = list[Cube[T]]


class Net:
    def __init__(
        self,
        layersize: list[int]  ,
        activfunc: list       ,
        weigvalue: Cube[float],
        biasvalue: Grid[float],
        currinput: Grid[int]  ,
        pastinput: Cube[int]  ,
        lifescore: int        ,
        lifelimit: int        ,
        goodmatch: bool       ,
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

    def forward(self, ten: Tensor) -> ...:
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
        goodmatch: None | bool        = None,
    ) -> None:
        super().__init__(
            layersize or defval.TNN_LAYERSIZE,
            activfunc or defval.TNN_ACTIVFUNC,
            weigvalue or defval.TNN_WEIGVALUE,
            biasvalue or defval.TNN_BIASVALUE,
            currinput or []                  ,
            pastinput or []                  ,
            lifescore or 42                  ,
            lifelimit or 0                   ,
            goodmatch or False               ,
        )

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
        goodmatch: None | bool        = None,
    ) -> None:
        super().__init__(
            layersize or defval.SNN_LAYERSIZE,
            activfunc or defval.SNN_ACTIVFUNC,
            weigvalue or defval.SNN_WEIGVALUE,
            biasvalue or defval.SNN_BIASVALUE,
            currinput or []                  ,
            pastinput or []                  ,
            lifescore or 42                  ,
            lifelimit or 0                   ,
            goodmatch or False               ,
        )

    def gen_layersize(self, elec: Elec) -> list[int]:
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
        goodmatch: None | bool        = None,
    ) -> None:
        super().__init__(
            layersize or defval.SNN_LAYERSIZE,
            activfunc or defval.SNN_ACTIVFUNC,
            weigvalue or defval.SNN_WEIGVALUE,
            biasvalue or defval.SNN_BIASVALUE,
            currinput or []                  ,
            pastinput or []                  ,
            lifescore or 42                  ,
            lifelimit or 0                   ,
            goodmatch or False               ,
        )

    def gen_activfunc(self, elec: Elec) -> list:
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
        goodmatch: None | bool        = None,
    ) -> None:
        super().__init__(
            layersize or defval.SNN_LAYERSIZE,
            activfunc or defval.SNN_ACTIVFUNC,
            weigvalue or defval.SNN_WEIGVALUE,
            biasvalue or defval.SNN_BIASVALUE,
            currinput or []                  ,
            pastinput or []                  ,
            lifescore or 42                  ,
            lifelimit or 0                   ,
            goodmatch or False               ,
        )

    def gen_weigvalue(self, elec: Elec) -> Cube[float]:
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
        goodmatch: None | bool        = None,
    ) -> None:
        super().__init__(
            layersize or defval.SNN_LAYERSIZE,
            activfunc or defval.SNN_ACTIVFUNC,
            weigvalue or defval.SNN_WEIGVALUE,
            biasvalue or defval.SNN_BIASVALUE,
            currinput or []                  ,
            pastinput or []                  ,
            lifescore or 42                  ,
            lifelimit or 0                   ,
            goodmatch or False               ,
        )

    def gen_biasvalue(self, elec: Elec) -> Grid[float]:
        ...


class Siamese(Net):
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
        goodmatch: None | bool        = None,
    ) -> None:
        super().__init__(
            layersize or defval.SNN_LAYERSIZE,
            activfunc or defval.SNN_ACTIVFUNC,
            weigvalue or defval.SNN_WEIGVALUE,
            biasvalue or defval.SNN_BIASVALUE,
            currinput or []                  ,
            pastinput or []                  ,
            lifescore or 42                  ,
            lifelimit or 0                   ,
            goodmatch or False               ,
        )

    def twins(self, curr: Grid[int], past: Grid[int]) -> bool:
        ...
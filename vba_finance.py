from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence


def DateSerial(y: int, m: int, d: int) -> date:
    """Python equivalent of VBA DateSerial with overflow behavior."""
    y_adj = y + (m - 1) // 12
    m_adj = (m - 1) % 12 + 1
    first = date(y_adj, m_adj, 1)
    return first + timedelta(days=d - 1)


def mati(c1_date: date, mode: int = 1, threshold_days: int | None = None) -> int:
    """Equivalent of VBA/Excel mati(C1,1) threshold in days."""
    if threshold_days is not None:
        return int(threshold_days)
    if mode != 1:
        return int((DateSerial(c1_date.year + 1, c1_date.month, c1_date.day) - c1_date).days)
    return int((DateSerial(c1_date.year + 1, c1_date.month, c1_date.day) - c1_date).days)


def _clean_curve_points(mt: Sequence[int | float], tx: Sequence[int | float]) -> tuple[list[int], list[float]]:
    # VBA style: points are consumed in order and stop at first 0 maturity sentinel.
    mt_nz: list[int] = []
    tx_nz: list[float] = []
    for m, r in zip(mt, tx):
        m_int = int(m)
        if m_int == 0:
            break
        if m_int < 0:
            continue
        mt_nz.append(m_int)
        tx_nz.append(float(r))
    return mt_nz, tx_nz


def calcul_taux(
    maturity: int,
    mt: Sequence[int | float],
    tx: Sequence[int | float],
    C1_date: date,
    mati_threshold_days: int | None = None,
) -> float:
    """Reproduce VBA calcul_taux structure and equations."""
    mt_nz, tx_nz = _clean_curve_points(mt, tx)
    d = len(mt_nz)
    if d == 0:
        raise ValueError("mt is empty / no valid maturities")
    if d == 1:
        return tx_nz[0]

    pivot = mati(C1_date, 1, mati_threshold_days)

    if maturity <= mt_nz[0]:
        return tx_nz[0]

    if maturity > mt_nz[0] and maturity <= mt_nz[-1]:
        i = 0
        A = mt_nz[i + 1]
        B = mt_nz[i]
        while maturity > A and (i + 1) < (d - 1):
            B = A
            i += 1
            A = mt_nz[i + 1]

        if (A <= pivot) or (B > pivot):
            return ((maturity - B) * (tx_nz[i + 1] - tx_nz[i]) / (A - B)) + tx_nz[i]

        if (A > pivot) and (B <= pivot):
            if maturity > pivot:
                di = C1_date + timedelta(days=mt_nz[i])
                Base = (di - DateSerial(di.year - 1, di.month, di.day)).days
                taux_equiv = ((1 + tx_nz[i] * B / 360.0) ** (Base / B)) - 1
                Z = ((maturity - B) * (tx_nz[i + 1] - taux_equiv) / (A - B)) + taux_equiv
                return Z
            di1 = C1_date + timedelta(days=mt_nz[i + 1])
            Base = (di1 - DateSerial(di1.year - 1, di1.month, di1.day)).days
            taux_equiv = (360.0 / A) * (((1 + tx_nz[i + 1]) ** (A / Base)) - 1)
            Z = ((maturity - B) * (taux_equiv - tx_nz[i]) / (A - B)) + tx_nz[i]
            return Z

        return ((maturity - B) * (tx_nz[i + 1] - tx_nz[i]) / (A - B)) + tx_nz[i]

    return (
        ((maturity - mt_nz[-2]) * (tx_nz[-1] - tx_nz[-2]) / (mt_nz[-1] - mt_nz[-2]))
        + tx_nz[-2]
    )


def cpz(maturity: int, duree: int, taux: Sequence[float]) -> float:
    """VBA cpz translation (zero-coupon bootstrapping by annual steps)."""
    if duree <= 0:
        return float(taux[0]) if taux else 0.0
    tzc = [0.0] * 30
    tzc[0] = float(taux[0])
    n = 2
    while n <= duree:
        somme = 0.0
        for i in range(1, n):
            somme += float(taux[n - 1]) / ((1.0 + tzc[i - 1]) ** i)
        tzc[n - 1] = (((1.0 + float(taux[n - 1])) / (1.0 - somme)) ** (1.0 / n)) - 1.0
        n += 1
    return tzc[n - 2]


def interpol(arg: bool, maturity: int, mtz: Sequence[int], txz: Sequence[float]) -> float:
    """VBA interpol translation. Active branch in your file is arg=True."""
    if not mtz or not txz or len(mtz) != len(txz):
        raise ValueError("mtz/txz invalid")

    if maturity <= int(mtz[0]):
        return float(txz[0])
    if maturity >= int(mtz[-1]):
        return float(txz[-1])

    i = 0
    A = int(mtz[i + 1])
    B = int(mtz[i])
    while maturity > A and i < len(mtz) - 2:
        B = A
        i += 1
        A = int(mtz[i + 1])
    P = ((maturity - B) * (float(txz[i + 1]) - float(txz[i])) / (A - B)) + float(txz[i])
    return float(P)


def conversion_actu_monnaitaire(
    arg: bool,
    maturity: int,
    date_flux: date,
    mt: Sequence[int | float],
    tx: Sequence[int | float],
    C1_date: date,
    mati_threshold_days: int | None = None,
) -> float:
    """VBA conversion_actu_monnaitaire translation."""
    taux = calcul_taux(maturity, mt, tx, C1_date, mati_threshold_days)
    if maturity == 0:
        return 0.0

    pivot = mati(C1_date, 1, mati_threshold_days)
    if (arg is False) and maturity <= pivot:
        di = C1_date + timedelta(days=maturity)
        alpha = (di - DateSerial(di.year - 1, di.month, di.day)).days
        return ((1.0 + taux * maturity / 360.0) ** (alpha / maturity)) - 1.0
    if (arg is False) and maturity > pivot:
        return taux
    if (arg is True) and maturity >= pivot:
        return (360.0 / maturity) * (((1.0 + taux) ** (maturity / 365.0)) - 1.0)
    return taux


def calcul_zerocp(
    maturity: int,
    date_flux: date,
    mt: Sequence[int | float],
    tx: Sequence[int | float],
    C1_date: date,
    mati_threshold_days: int | None = None,
) -> float:
    """VBA calcul_zerocp translation."""
    pivot = mati(C1_date, 1, mati_threshold_days)
    if maturity <= pivot:
        return conversion_actu_monnaitaire(False, maturity, date_flux, mt, tx, C1_date, mati_threshold_days)

    # Build annual maturity ladder (1..30 years) as in VBA.
    duree = [i + 1 for i in range(30)]
    matu = [
        (DateSerial(C1_date.year + d, C1_date.month, C1_date.day) - C1_date).days
        for d in duree
    ]

    j = 0
    A = matu[j + 1]
    B = matu[j]
    while maturity > A and j < len(matu) - 2:
        B = A
        j += 1
        A = matu[j + 1]

    tzc = [0.0] * 30
    taux = [0.0] * 30
    tzc[0] = conversion_actu_monnaitaire(False, pivot, date_flux, mt, tx, C1_date, mati_threshold_days)
    taux[0] = tzc[0]

    for k in range(1, j + 2):
        taux[k] = calcul_taux(matu[k], mt, tx, C1_date, mati_threshold_days)
        tzc[k] = cpz(matu[k], duree[k], taux)

    return interpol(True, maturity, matu, tzc)


def DatePr_Cp(datejouissance: date, datevaleur: date) -> date:
    """Equivalent of VBA DatePr_Cp loop."""
    i = 0
    while DateSerial(datejouissance.year + i, datejouissance.month, datejouissance.day) <= datevaleur:
        i += 1

    if i == 0:
        return DateSerial(datejouissance.year + 1, datejouissance.month, datejouissance.day)
    return DateSerial(datejouissance.year + i, datejouissance.month, datejouissance.day)


@dataclass
class AmortissableResult:
    prix: float
    tzcpp: float
    datefl: list[date]
    amort: list[float]
    crd: list[float]
    cpn: list[float]


def prix_amortissable(
    date_valeur: date,
    date_emission: date,
    date_echeance: date,
    date_jouissance: date,
    nominal: int,
    tf: float,
    spread: float,
    nbramort: int,
    mt: Sequence[int | float] | None = None,
    tx: Sequence[int | float] | None = None,
    c1_date: date | None = None,
    mati_threshold_days: int | None = None,
) -> AmortissableResult:
    """Direct translation of active VBA blocks in prix_amortissable."""
    if nbramort <= 0:
        raise ValueError("nbramort must be > 0")

    if c1_date is None:
        c1_date = date_valeur
    if mt is None:
        mt = []
    if tx is None:
        tx = []

    datefl: list[date] = [date(1900, 1, 1)] * max(101, nbramort + 1)
    amort: list[float] = [0.0] * max(101, nbramort + 1)
    crd: list[float] = [0.0] * max(101, nbramort + 1)
    cpn: list[float] = [0.0] * max(101, nbramort + 1)
    tzcpp: list[float] = [0.0] * max(101, nbramort + 1)
    fract: list[float] = [0.0] * max(101, nbramort + 1)
    fluxvl: list[float] = [0.0] * max(101, nbramort + 1)

    datefl[0] = DateSerial(date_emission.year + 1, date_emission.month, date_emission.day)
    amort[0] = round(nominal / nbramort, 2)
    crd[0] = float(nominal)
    cpn[0] = round(crd[0] * tf, 2)

    for i in range(1, nbramort):
        datefl[i] = DateSerial(datefl[i - 1].year + 1, datefl[i - 1].month, datefl[i - 1].day)
        if datefl[i] != date_echeance:
            amort[i] = round(nominal / nbramort, 2)
        else:
            amort[i] = crd[i - 1] - amort[i - 1]
        crd[i] = crd[i - 1] - amort[i]
        cpn[i] = round(crd[i] * tf, 2)

    j = 1
    while DateSerial(date_emission.year + j, date_emission.month, date_emission.day) <= date_valeur:
        j += 1

    m0 = (datefl[j - 1] - date_valeur).days
    tzcpp[j - 1] = (
        round(calcul_zerocp(m0, datefl[j - 1], mt, tx, c1_date, mati_threshold_days), 5) + spread
    )
    base0 = (datefl[j - 1] - DateSerial(datefl[j - 1].year - 1, datefl[j - 1].month, datefl[j - 1].day)).days
    fract[j - 1] = (datefl[j - 1] - date_valeur).days / base0
    fluxvl[j - 1] = (amort[j - 1] + cpn[j - 1]) / ((1 + tzcpp[j - 1]) ** fract[j - 1])
    P = fluxvl[j - 1]

    for i in range(j, nbramort):
        m_i = (datefl[i] - date_valeur).days
        tzcpp[i] = (
            round(calcul_zerocp(m_i, datefl[i], mt, tx, c1_date, mati_threshold_days), 5) + spread
        )
        fract[i] = fract[i - 1] + 1
        fluxvl[i] = (amort[i] + cpn[i]) / ((1 + tzcpp[i]) ** fract[i])
        P += fluxvl[i]

    return AmortissableResult(
        prix=P,
        tzcpp=tzcpp[j - 1] if j - 1 >= 0 else 0.0,
        datefl=datefl[:nbramort],
        amort=amort[:nbramort],
        crd=crd[:nbramort],
        cpn=cpn[:nbramort],
    )

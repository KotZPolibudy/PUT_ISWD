# PUT_ISWD
Inteligentne Systemy Wspomagania Decyzji


Projekt 1-2 osobowy z terminem na 18 czerwca z wymaganiami: ????

### Projekt 2
Nie wiem dlaczego, ale po każdym resecie cmd python zapomina o istnieniu paczki, więc koniecznie przed odpaleniem (dla standardowej ścieżki instalacyjnej pygraphviz):

python -m pip install --config-settings="--global-option=build_ext" `
              --config-settings="--global-option=-IC:\Program Files\Graphviz\include" `
              --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" `
              pygraphviz

cd projekt2
python -m electre_iii_pl.main ./electre_iii_pl/data/lecture
python -m promethee_pl.main ./promethee_pl/data/lecture

Jeżeli chcesz wykonać na swoim zbiorze danych przygotowanym (z określonymi kryteriami dla wybranych kolumn i określonym ziarnie losowania):
cd projekt2
python create_dataset.py
python -m electre_iii_pl.main ./electre_iii_pl/data/own
python -m promethee_pl.main ./promethee_pl/data/own

lub dla prostoty:
cd projekt2
python create_dataset.py; python -m electre_iii_pl.main ./electre_iii_pl/data/own; python -m promethee_pl.main ./promethee_pl/data/own

Preferencje (ich progi i wagi kryteriów) są ustalone dla całego zbioru i nie wymagają zmian.
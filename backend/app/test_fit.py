import matplotlib.pyplot as plt

# Dane wpisane na sztywno
n_starts = [2, 5, 10, 20, 40, 80]
K2 = [0.2110859794, 0.2111, 0.20826314, 0.20726314, 0.2051693133, 0.205136166]

# Mała poprawka błędów w tabeli (brakujące zera w oryginalnym obrazku na pozycjach dla 10 i 40)
# Jeśli to nie były błędy w zaokrągleniu, odkomentuj poniższą linijkę:
# K2 = [0.21108598, 0.2111, 0.2082, 0.20726314, 0.20516931, 0.20513617]

# Tworzenie wykresu
plt.figure(figsize=(8, 5))
plt.plot(n_starts, K2, marker='o', linestyle='-', color='b', label='K2')

# Tytuł i opisy osi
plt.title('Wykres liniowy K2 w zależności od n_starts')
plt.xlabel('n_starts')
plt.ylabel('K2')

# Dodanie siatki dla lepszej czytelności
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Wyświetlenie wykresu
plt.show()
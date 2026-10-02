"""
app/sensors/machine_state_simulator.py

Simule l'état marche/arrêt de la machine, en l'absence d'accès direct aux
capteurs/PLC du site OCP dans le délai du POC (cf. cahier des charges,
section 5 et 12 : "l'état marche/arrêt de la machine sera simulé").

Principe de précaution (aligné sur le standard S-HSE-22, consignation) :
    "Toute installation est considérée en service et donc dangereuse, tant
    que la vérification de l'élimination des sources d'énergie n'a pas été
    réalisée."
Donc par défaut, si l'état n'est pas explicitement confirmé "arrêt", on
considère la machine comme EN MARCHE (comportement prudent).
"""


class MachineStateSimulator:
    """
    Simule l'état d'une ou plusieurs machines. Pour le POC, l'état est fixé
    manuellement (pas de vrai capteur/PLC connecté) — mais l'interface est
    conçue pour qu'un vrai capteur puisse remplacer cette simulation plus
    tard sans changer le reste du code (même méthode is_running()).

    Utilisation simple (une seule machine, état global) :
        simulator = MachineStateSimulator(default_state="marche")
        simulator.is_running()  # True

    Utilisation avancée (état différent par zone/machine) :
        simulator.set_state("convoyeur_A", "arret")
        simulator.is_running("convoyeur_A")  # False
        simulator.is_running("convoyeur_B")  # True (défaut, non confirmé arrêté)
    """

    VALID_STATES = {"marche", "arret"}

    def __init__(self, default_state: str = "marche"):
        """
        Args:
            default_state: état par défaut ("marche" ou "arret") appliqué
                à toute machine dont l'état n'a pas été fixé explicitement.
                "marche" par défaut, conformément au principe de précaution
                S-HSE-22 (on ne présume jamais un arrêt sécurisé sans preuve).
        """
        if default_state not in self.VALID_STATES:
            raise ValueError(f"default_state doit être 'marche' ou 'arret', reçu : {default_state}")
        self.default_state = default_state
        self._states: dict[str, str] = {}

    def set_state(self, machine_id: str, state: str):
        """Fixe manuellement l'état d'une machine identifiée par machine_id."""
        if state not in self.VALID_STATES:
            raise ValueError(f"state doit être 'marche' ou 'arret', reçu : {state}")
        self._states[machine_id] = state

    def is_running(self, machine_id: str = "default") -> bool:
        """
        Retourne True si la machine est considérée en marche.
        Si aucun état n'a été fixé pour ce machine_id, retourne l'état par
        défaut (prudence : "marche" sauf configuration contraire).
        """
        state = self._states.get(machine_id, self.default_state)
        return state == "marche"

    def get_state_label(self, machine_id: str = "default") -> str:
        """Retourne l'état lisible ('marche' ou 'arret') pour affichage."""
        return self._states.get(machine_id, self.default_state)

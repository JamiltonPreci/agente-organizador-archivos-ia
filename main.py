import time
from collections import deque
import heapq


# 1. ESTRUCTURA Y CLASES BASE DE AIMA (Problem y Node)


class Problem:
    """Clase base para definir un problema de busqueda."""
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        if isinstance(self.goal, list):
            return state in self.goal
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1


class Node:
    """Clase Nodo para construir el arbol de busqueda."""
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return f"<Node {self.state}>"

    def __lt__(self, node):
        return self.path_cost < node.path_cost

    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        next_cost = problem.path_cost(self.path_cost, self.state, action, next_state)
        return Node(next_state, self, action, next_cost)

    def path_tree(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


# 2. DEFINICIÓN DEL PROBLEMA CAMPUSGUARD


class CampusGuardProblem(Problem):
    """
    Estado: tupla (Riesgo, Inasistencias, Intervencion)
    """
    def __init__(self, initial=('Alto', 35, 'SinAccion'), goal=('Bajo', 20, 'Recuperado')):
        super().__init__(initial, goal)

    def actions(self, state):
        riesgo, inasist, interv = state
        accs = []
        if riesgo == 'Alto':
            accs.append('Assign_Tutoring')
            accs.append('Notify_Wellness')
        elif riesgo == 'Medio':
            accs.append('Adjust_Load')
            accs.append('Assign_Tutoring')
        elif riesgo == 'Bajo' and interv != 'Recuperado':
            accs.append('Evaluate_Progress')
        return accs

    def result(self, state, action):
        riesgo, inasist, interv = state
        
        if action == 'Assign_Tutoring':
            return ('Medio', max(0, inasist - 5), 'TutoriaAsignada')
        elif action == 'Notify_Wellness':
            return ('Medio', max(0, inasist - 10), 'BienestarNotificado')
        elif action == 'Adjust_Load':
            return ('Bajo', max(0, inasist - 15), 'CargaReducida')
        elif action == 'Evaluate_Progress':
            return ('Bajo', inasist, 'Recuperado')
        
        return state

    def goal_test(self, state):
        riesgo, inasist, interv = state
        return riesgo == 'Bajo' and interv == 'Recuperado'

    def path_cost(self, c, state1, action, state2):
        costs = {
            'Assign_Tutoring': 12,
            'Notify_Wellness': 24,
            'Adjust_Load': 6,
            'Evaluate_Progress': 2
        }
        return c + costs.get(action, 1)


# Heurística admisible para A* (horas estimadas de intervención pendientes)
def heuristica_campus_guard(node):
    riesgo, inasist, interv = node.state
    h = 0
    if riesgo == 'Alto':
        h += 14  # Estimación admisible (12 + 2)
    elif riesgo == 'Medio':
        h += 8   # Estimación admisible (6 + 2)
    elif riesgo == 'Bajo' and interv != 'Recuperado':
        h += 2
    return h


# 3. ALGORITMOS DE BÚSQUEDA (BFS, DFS, A*)


def breadth_first_graph_search(problem):
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node, 0
    frontier = deque([node])
    explored = set()
    nodes_expanded = 0

    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        nodes_expanded += 1
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child, nodes_expanded
                frontier.append(child)
    return None, nodes_expanded


def depth_first_graph_search(problem):
    frontier = [Node(problem.initial)]
    explored = set()
    nodes_expanded = 0

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node, nodes_expanded
        explored.add(node.state)
        nodes_expanded += 1
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
    return None, nodes_expanded


def astar_search(problem, h=heuristica_campus_guard):
    node = Node(problem.initial)
    frontier = []
    heapq.heappush(frontier, (node.path_cost + h(node), node))
    explored = set()
    nodes_expanded = 0

    while frontier:
        f, node = heapq.heappop(frontier)
        if problem.goal_test(node.state):
            return node, nodes_expanded
        explored.add(node.state)
        nodes_expanded += 1
        for child in node.expand(problem):
            if child.state not in explored:
                heapq.heappush(frontier, (child.path_cost + h(child), child))
    return None, nodes_expanded


# 4. EJECUCIÓN Y TABLA DE RESULTADOS


if __name__ == '__main__':
    problem = CampusGuardProblem()
    algoritmos = [
        ("BFS (Anchura)", breadth_first_graph_search),
        ("DFS (Profundidad)", depth_first_graph_search),
        ("A* (Estrella)", astar_search)
    ]

    print("\n" + "=" * 80)
    print(f"{'Algoritmo':<20} | {'Nodos Exp.':<12} | {'Tiempo (ms)':<12} | {'Costo Total (h)':<15} | {'Óptimo':<8}")
    print("=" * 80)

    for nombre, func in algoritmos:
        t0 = time.perf_counter()
        sol_node, exp_nodes = func(problem)
        t1 = time.perf_counter()
        
        tiempo_ms = (t1 - t0) * 1000
        costo = sol_node.path_cost if sol_node else "N/A"
        es_optimo = "Sí" if nombre != "DFS (Profundidad)" else "No"

        print(f"{nombre:<20} | {exp_nodes:<12} | {tiempo_ms:<12.4f} | {costo:<15} | {es_optimo:<8}")

    print("=" * 80 + "\n")
    
    if sol_node:
        print("Camino óptimo encontrado por A*:")
        for i, step in enumerate(sol_node.path_tree()):
            print(f"  Paso {i}: Estado = {step.state} | Acción = {step.action} | Costo Acumulado = {step.path_cost} horas")
        print("\n")
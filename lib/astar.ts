// Graph representation using adjacency list
type Graph = {
  [key: string]: {
    [key: string]: number
  }
}

// Node with coordinates for heuristic calculation
type NodeWithCoords = {
  id: string
  lat: number
  lng: number
}

// Priority queue for A* algorithm
class PriorityQueue {
  private elements: { element: string; priority: number }[]

  constructor() {
    this.elements = []
  }

  enqueue(element: string, priority: number): void {
    this.elements.push({ element, priority })
    this.elements.sort((a, b) => a.priority - b.priority)
  }

  dequeue(): { element: string; priority: number } {
    return this.elements.shift()!
  }

  isEmpty(): boolean {
    return this.elements.length === 0
  }

  contains(element: string): boolean {
    return this.elements.some((item) => item.element === element)
  }
}

/**
 * Calculate the Euclidean distance between two nodes (heuristic function)
 * @param a - First node with coordinates
 * @param b - Second node with coordinates
 * @returns The Euclidean distance
 */
function euclideanDistance(a: NodeWithCoords, b: NodeWithCoords): number {
  return Math.sqrt(Math.pow(a.lat - b.lat, 2) + Math.pow(a.lng - b.lng, 2))
}

/**
 * Implementation of A* algorithm for finding the shortest path
 * @param graph - The graph represented as an adjacency list
 * @param startNode - The starting node
 * @param endNode - The destination node
 * @param nodeCoords - Map of node IDs to their coordinates
 * @returns Array of nodes representing the shortest path
 */
export function aStar(
  graph: Graph,
  startNode: string,
  endNode: string,
  nodeCoords: { [key: string]: { lat: number; lng: number } },
): string[] {
  // Priority queue for nodes to visit
  const openSet = new PriorityQueue()
  openSet.enqueue(startNode, 0)

  // Track where we came from
  const cameFrom: { [key: string]: string | null } = {}

  // Cost from start to node
  const gScore: { [key: string]: number } = {}
  for (const node in graph) {
    gScore[node] = Number.POSITIVE_INFINITY
  }
  gScore[startNode] = 0

  // Estimated total cost from start to goal through node
  const fScore: { [key: string]: number } = {}
  for (const node in graph) {
    fScore[node] = Number.POSITIVE_INFINITY
  }

  // Calculate initial fScore using heuristic
  const startCoords: NodeWithCoords = {
    id: startNode,
    ...nodeCoords[startNode],
  }

  const endCoords: NodeWithCoords = {
    id: endNode,
    ...nodeCoords[endNode],
  }

  fScore[startNode] = euclideanDistance(startCoords, endCoords)

  while (!openSet.isEmpty()) {
    const current = openSet.dequeue().element

    if (current === endNode) {
      return reconstructPath(cameFrom, current)
    }

    for (const neighbor in graph[current]) {
      // Distance from start to neighbor through current
      const tentativeGScore = gScore[current] + graph[current][neighbor]

      if (tentativeGScore < gScore[neighbor]) {
        // This path is better than any previous one
        cameFrom[neighbor] = current
        gScore[neighbor] = tentativeGScore

        const neighborCoords: NodeWithCoords = {
          id: neighbor,
          ...nodeCoords[neighbor],
        }

        fScore[neighbor] = gScore[neighbor] + euclideanDistance(neighborCoords, endCoords)

        if (!openSet.contains(neighbor)) {
          openSet.enqueue(neighbor, fScore[neighbor])
        }
      }
    }
  }

  // No path found
  return []
}

/**
 * Reconstruct the path from the cameFrom map
 * @param cameFrom - Map of nodes to their predecessors
 * @param current - The end node
 * @returns The reconstructed path
 */
function reconstructPath(cameFrom: { [key: string]: string | null }, current: string): string[] {
  const path = [current]
  while (cameFrom[current]) {
    current = cameFrom[current]!
    path.unshift(current)
  }
  return path
}

/**
 * Apply traffic conditions to a graph by modifying edge weights
 * @param graph - The original graph
 * @param trafficLevel - Traffic level (low, medium, high)
 * @returns A new graph with adjusted weights
 */
export function applyTraffic(graph: Graph, trafficLevel: "low" | "medium" | "high"): Graph {
  const trafficGraph: Graph = JSON.parse(JSON.stringify(graph)) // Deep copy

  // Traffic multipliers
  const multipliers = {
    low: 1.1,
    medium: 1.5,
    high: 2.5,
  }

  const multiplier = multipliers[trafficLevel]

  // Apply random traffic to some edges
  for (const node in trafficGraph) {
    for (const neighbor in trafficGraph[node]) {
      // Apply traffic multiplier with some randomness
      const randomFactor = 0.5 + Math.random()
      trafficGraph[node][neighbor] *= multiplier * randomFactor
    }
  }

  return trafficGraph
}


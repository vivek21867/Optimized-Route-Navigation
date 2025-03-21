// Graph representation using adjacency list
type Graph = {
  [key: string]: {
    [key: string]: number
  }
}

/**
 * Implementation of Dijkstra's algorithm for finding the shortest path
 * @param graph - The graph represented as an adjacency list
 * @param startNode - The starting node
 * @param endNode - The destination node
 * @returns Array of nodes representing the shortest path
 */
export function dijkstra(graph: Graph, startNode: string, endNode: string): string[] {
  // Set distances for all nodes to Infinity except start node
  const distances: { [key: string]: number } = {}
  const previous: { [key: string]: string | null } = {}
  const nodes = new Set<string>()

  // Initialize data structures
  for (const vertex in graph) {
    distances[vertex] = Number.POSITIVE_INFINITY
    previous[vertex] = null
    nodes.add(vertex)
  }
  distances[startNode] = 0

  while (nodes.size > 0) {
    // Find node with minimum distance
    let minNode: string | null = null
    for (const node of nodes) {
      if (minNode === null || distances[node] < distances[minNode]) {
        minNode = node
      }
    }

    // If we reached the end node or no path exists
    if (minNode === endNode || distances[minNode] === Number.POSITIVE_INFINITY) {
      break
    }

    nodes.delete(minNode)

    // Update distances to neighbors
    for (const neighbor in graph[minNode]) {
      const alt = distances[minNode] + graph[minNode][neighbor]
      if (alt < distances[neighbor]) {
        distances[neighbor] = alt
        previous[neighbor] = minNode
      }
    }
  }

  // Reconstruct path
  const path: string[] = []
  let current: string | null = endNode

  while (current !== null) {
    path.unshift(current)
    current = previous[current]
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


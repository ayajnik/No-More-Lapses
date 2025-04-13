from crewai_tools import CSVSearchTool
import pandas as pd

class ChunkedCSVSearchTool:
    def __init__(self, csv_path: str, chunk_size: int = 1000):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.chunk_tools = self._create_chunked_tools()

    def _create_chunked_tools(self):
        """Split the CSV into chunked tools for scalable querying."""
        tools = []
        try:
            
            for i, chunk in enumerate(pd.read_csv(self.csv_path, chunksize=self.chunk_size)):
                chunk_path = self.csv_path.replace('.csv', f'_chunk_{i}.csv')
                chunk.to_csv(chunk_path, index=False)
                tools.append(CSVSearchTool(csv_path=chunk_path))
        except Exception as e:
            print(f"Error while chunking CSV: {e}")
        return tools

    def search(self, query: str, top_k: int = 3):
        """Search across all chunked tools and return aggregated results."""
        results = []
        for tool in self.chunk_tools:
            try:
                result = tool.search(query, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"Search failed in a chunk: {e}")
        return "\n".join(results)

    def as_tool(self):
        """Returns a callable search tool for CrewAI agent compatibility."""
        return {
            "name": "ChunkedCSVSearchTool",
            "description": "Searches across chunked CSV data for lapse prediction analysis.",
            "function": lambda query: self.search(query)
        }

if __name__ == "__main__":

    csv_tool_wrapper = ChunkedCSVSearchTool(
        csv_path="lapse_researcher/src/lapse_researcher/artifacts/lapse_predictions.csv",
        chunk_size=1000
    )

    search_tool = csv_tool_wrapper.as_tool() 

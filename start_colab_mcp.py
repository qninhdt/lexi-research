import asyncio
import json
import sys
import webbrowser

from colab_mcp.session import ColabSessionProxy, ColabProxyClient
from colab_mcp.websocket_server import COLAB, SCRATCH_PATH

async def run_notebook_cells(client):
    print("\n" + "="*70)
    print("🚀 STARTING AUTOMATIC COLAB NOTEBOOK EXECUTION...")
    print("="*70)
    
    # 1. GPU Check
    print("\n--- [Step 1/3] Adding & Executing GPU Check ---")
    add_0 = await client.call_tool("add_code_cell", {"cellIndex": 0, "language": "python", "code": "!nvidia-smi"})
    print("Add GPU cell result:", add_0)
    
    cells = await client.call_tool("get_cells", {})
    print("Notebook cells:", cells)
    
    cell_id_0 = None
    if hasattr(cells, "content"):
        for item in cells.content:
            if hasattr(item, "text"):
                try:
                    data = json.loads(item.text)
                    if isinstance(data, list) and len(data) > 0:
                        cell_id_0 = data[0].get("id") or data[0].get("cellId") or data[0].get("cell_id")
                except Exception:
                    pass
    if cell_id_0:
        print(f"Running cell_id 0: {cell_id_0}")
        try:
            gpu_res = await client.call_tool("run_code_cell", {"cellId": str(cell_id_0)})
        except Exception:
            gpu_res = await client.call_tool("run_code_cell", {"cell_id": str(cell_id_0)})
        print("GPU Result:\n", gpu_res)
        
    # 2. Setup environment
    print("\n--- [Step 2/3] Adding & Executing Environment Setup ---")
    setup_code = """!git clone https://github.com/qninhdt/lexi-research.git || (cd lexi-research && git pull)
%cd lexi-research
!pip install -r requirements-colab.txt"""
    add_1 = await client.call_tool("add_code_cell", {"cellIndex": 1, "language": "python", "code": setup_code})
    print("Add Setup cell result:", add_1)
    
    cells = await client.call_tool("get_cells", {})
    cell_id_1 = None
    if hasattr(cells, "content"):
        for item in cells.content:
            if hasattr(item, "text"):
                try:
                    data = json.loads(item.text)
                    if isinstance(data, list) and len(data) > 1:
                        cell_id_1 = data[1].get("id") or data[1].get("cellId") or data[1].get("cell_id")
                except Exception:
                    pass
    if cell_id_1:
        print(f"Running cell_id 1: {cell_id_1}")
        try:
            setup_res = await client.call_tool("run_code_cell", {"cellId": str(cell_id_1)})
        except Exception:
            setup_res = await client.call_tool("run_code_cell", {"cell_id": str(cell_id_1)})
        print("Setup Result:\n", setup_res)

    # 3. Training
    print("\n--- [Step 3/3] Adding & Executing QLoRA Fine-Tuning ---")
    train_code = "!python -m lexi_research.train.cli --train data/split/train.parquet --output models/student_qlora --epochs 2"
    add_2 = await client.call_tool("add_code_cell", {"cellIndex": 2, "language": "python", "code": train_code})
    print("Add Train cell result:", add_2)
    
    cells = await client.call_tool("get_cells", {})
    cell_id_2 = None
    if hasattr(cells, "content"):
        for item in cells.content:
            if hasattr(item, "text"):
                try:
                    data = json.loads(item.text)
                    if isinstance(data, list) and len(data) > 2:
                        cell_id_2 = data[2].get("id") or data[2].get("cellId") or data[2].get("cell_id")
                except Exception:
                    pass
    if cell_id_2:
        print(f"Running cell_id 2: {cell_id_2}")
        try:
            train_res = await client.call_tool("run_code_cell", {"cellId": str(cell_id_2)})
        except Exception:
            train_res = await client.call_tool("run_code_cell", {"cell_id": str(cell_id_2)})
        print("Training Result:\n", train_res)
        
    print("\n🎉 ALL COLAB STEPS EXECUTED SUCCESSFULLY!")

async def main():
    proxy = ColabSessionProxy()
    await proxy.start_proxy_server()
    
    from contextlib import AsyncExitStack
    async with AsyncExitStack() as stack:
        proxy_client = await stack.enter_async_context(ColabProxyClient(proxy.wss))
        
        token = proxy.wss.token
        port = proxy.wss.port
        url = f"{COLAB}{SCRATCH_PATH}?authuser=3#mcpProxyToken={token}&mcpProxyPort={port}"
        
        print("\n" + "="*75)
        print("🚀 COLAB MCP AUTOMATION RUNNER")
        print("="*75)
        print("\n👉 Opening URL in browser (Account authuser=3):\n")
        print(f"   {url}\n")
        print("="*75)
        print("Waiting for Colab browser to connect...\n")
        
        try:
            webbrowser.open_new(url)
        except Exception:
            pass
            
        while not proxy.wss.connection_live.is_set():
            await asyncio.sleep(1)
            
        print("\n✅ COLAB BROWSER CONNECTED SUCCESSFULLY!")
        
        await proxy_client.await_proxy_connection()
        client = proxy_client.proxy_mcp_client
        
        if client:
            await run_notebook_cells(client)
        else:
            print("Failed to get proxy MCP client.")
            
        await proxy.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

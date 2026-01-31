from app.api.v1.endpoints.memory_syscall import router

print(f"Router prefix: {router.prefix}")
print(f"Routes ({len(router.routes)}):")
for i, route in enumerate(router.routes):
    path = route.path if hasattr(route, 'path') else 'N/A'
    methods = route.methods if hasattr(route, 'methods') else 'N/A'
    print(f"  {i}: {path} - {methods}")

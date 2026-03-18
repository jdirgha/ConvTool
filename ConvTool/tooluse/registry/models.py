"""Domain models for the tool registry.

All tool metadata is represented as frozen dataclasses to enforce
immutability once the registry is constructed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParameterSchema:
    name: str
    type: str
    description: str
    required: bool
    default: str | int | float | None = None
    enum: list[str] | None = None


@dataclass(frozen=True)
class ResponseField:
    name: str
    type: str


@dataclass(frozen=True)
class Endpoint:
    id: str
    name: str
    tool_name: str
    description: str
    method: str
    url: str
    parameters: list[ParameterSchema]
    response_fields: list[ResponseField]

    @property
    def required_parameters(self) -> list[ParameterSchema]:
        return [p for p in self.parameters if p.required]

    @property
    def optional_parameters(self) -> list[ParameterSchema]:
        return [p for p in self.parameters if not p.required]


@dataclass(frozen=True)
class ConceptTag:
    name: str


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    category: str
    endpoints: list[Endpoint]
    concepts: list[ConceptTag]


@dataclass
class ToolRegistry:
    tools: list[Tool] = field(default_factory=list)

    # Lazily built indices for O(1) lookup — invalidated when tools are mutated.
    _tool_index: dict[str, Tool] = field(default_factory=dict, repr=False, compare=False)
    _endpoint_index: dict[str, Endpoint] = field(default_factory=dict, repr=False, compare=False)

    @property
    def all_endpoints(self) -> list[Endpoint]:
        return [ep for tool in self.tools for ep in tool.endpoints]

    def get_tool(self, name: str) -> Tool | None:
        if not self._tool_index:
            self._tool_index = {t.name: t for t in self.tools}
        return self._tool_index.get(name)

    def get_endpoint(self, endpoint_id: str) -> Endpoint | None:
        if not self._endpoint_index:
            self._endpoint_index = {ep.id: ep for ep in self.all_endpoints}
        return self._endpoint_index.get(endpoint_id)

    def to_dict(self) -> dict:
        """Serialize registry to a JSON-compatible dict."""
        def _param(p: ParameterSchema) -> dict:
            d: dict = {
                "name": p.name,
                "type": p.type,
                "description": p.description,
                "required": p.required,
            }
            if p.default is not None:
                d["default"] = p.default
            if p.enum is not None:
                d["enum"] = p.enum
            return d

        def _endpoint(ep: Endpoint) -> dict:
            return {
                "id": ep.id,
                "name": ep.name,
                "tool_name": ep.tool_name,
                "description": ep.description,
                "method": ep.method,
                "url": ep.url,
                "parameters": [_param(p) for p in ep.parameters],
                "response_fields": [
                    {"name": rf.name, "type": rf.type}
                    for rf in ep.response_fields
                ],
            }

        def _tool(t: Tool) -> dict:
            return {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "endpoints": [_endpoint(ep) for ep in t.endpoints],
                "concepts": [c.name for c in t.concepts],
            }

        return {"tools": [_tool(t) for t in self.tools]}

    @classmethod
    def from_dict(cls, data: dict) -> ToolRegistry:
        """Deserialize registry from a dict (inverse of to_dict)."""
        tools: list[Tool] = []
        for td in data.get("tools", []):
            endpoints: list[Endpoint] = []
            for epd in td.get("endpoints", []):
                params = [
                    ParameterSchema(
                        name=p["name"],
                        type=p["type"],
                        description=p.get("description", ""),
                        required=p.get("required", False),
                        default=p.get("default"),
                        enum=p.get("enum"),
                    )
                    for p in epd.get("parameters", [])
                ]
                resp_fields = [
                    ResponseField(name=rf["name"], type=rf["type"])
                    for rf in epd.get("response_fields", [])
                ]
                endpoints.append(Endpoint(
                    id=epd["id"],
                    name=epd["name"],
                    tool_name=epd["tool_name"],
                    description=epd.get("description", ""),
                    method=epd.get("method", "GET"),
                    url=epd.get("url", ""),
                    parameters=params,
                    response_fields=resp_fields,
                ))
            concepts = [ConceptTag(name=c) for c in td.get("concepts", [])]
            tools.append(Tool(
                name=td["name"],
                description=td.get("description", ""),
                category=td.get("category", ""),
                endpoints=endpoints,
                concepts=concepts,
            ))
        return cls(tools=tools)

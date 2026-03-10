"use client";

import { useDeferredValue, useState } from "react";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable
} from "@tanstack/react-table";
import { ArrowUpDown, Search } from "lucide-react";
import { DetailModal } from "@/components/dashboard/detail-modal";
import { cn, titleCase } from "@/lib/utils";

type DataTableProps<TData extends Record<string, unknown>> = {
  title: string;
  subtitle: string;
  data: TData[];
  columns: ColumnDef<TData>[];
};

export function DataTable<TData extends Record<string, unknown>>({
  title,
  subtitle,
  data,
  columns
}: DataTableProps<TData>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [search, setSearch] = useState("");
  const [selectedRow, setSelectedRow] = useState<TData | null>(null);
  const deferredSearch = useDeferredValue(search);

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      globalFilter: deferredSearch
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: (row, _columnId, filterValue) => {
      const value = filterValue.toLowerCase();
      return Object.values(row.original).some((field) =>
        String(field ?? "")
          .toLowerCase()
          .includes(value)
      );
    }
  });

  const rowCountLabel = `${table.getFilteredRowModel().rows.length} rows`;

  return (
    <>
      <section className="surface rounded-[30px] border border-[color:var(--color-line)] p-5">
        <div className="mb-4 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h3 className="font-display text-[1.7rem] tracking-[-0.05em] text-ink">{title}</h3>
            <p className="mt-1 text-sm text-muted">{subtitle}</p>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <label className="relative block min-w-[220px]">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
              <input
                className="w-full rounded-full border border-[color:var(--color-line)] bg-slate-950/55 py-2.5 pl-10 pr-4 text-sm text-ink outline-none transition focus:border-[color:var(--color-line-strong)]"
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search rows"
                value={search}
              />
            </label>
            <div className="rounded-full border border-[color:var(--color-line)] px-4 py-2 text-xs uppercase tracking-[0.2em] text-muted">
              {rowCountLabel}
            </div>
          </div>
        </div>

        <div className="thin-scrollbar overflow-auto rounded-[24px] border border-white/5 bg-slate-950/42">
          <table className="min-w-full border-collapse text-left">
            <thead>
              {table.getHeaderGroups().map((headerGroup) => (
                <tr key={headerGroup.id} className="border-b border-white/5">
                  {headerGroup.headers.map((header) => (
                    <th
                      className="whitespace-nowrap px-4 py-3 text-[11px] uppercase tracking-[0.18em] text-muted"
                      key={header.id}
                    >
                      {header.isPlaceholder ? null : (
                        <button
                          className={cn(
                            "inline-flex items-center gap-2 transition hover:text-ink",
                            header.column.getCanSort() ? "cursor-pointer" : "cursor-default"
                          )}
                          onClick={header.column.getToggleSortingHandler()}
                          type="button"
                        >
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          {header.column.getCanSort() ? <ArrowUpDown className="h-3.5 w-3.5" /> : null}
                        </button>
                      )}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr
                  className="cursor-pointer border-b border-white/5 text-sm text-slate-100 transition hover:bg-white/[0.03]"
                  key={row.id}
                  onClick={() => setSelectedRow(row.original)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <td className="px-4 py-3 align-top text-sm text-slate-100/90" key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {!table.getRowModel().rows.length ? (
            <div className="px-4 py-12 text-center text-sm text-muted">No rows matched the current filter.</div>
          ) : null}
        </div>
      </section>

      <DetailModal
        onClose={() => setSelectedRow(null)}
        open={selectedRow !== null}
        payload={selectedRow}
        title={selectedRow ? `${titleCase(title)} Detail` : title}
      />
    </>
  );
}

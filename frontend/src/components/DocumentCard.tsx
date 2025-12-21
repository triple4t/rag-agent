import { FileText, Trash2, Layers } from 'lucide-react';
import type { Document } from '@/types';

interface DocumentCardProps {
  document: Document;
  onDelete: (id: string) => void;
}

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const DocumentCard = ({ document, onDelete }: DocumentCardProps) => {
  return (
    <div className="group flex items-start gap-3 p-3 rounded-lg hover:bg-sidebar-border/50 transition-colors animate-slide-in-left">
      <div className="w-10 h-10 rounded-lg bg-accent flex items-center justify-center flex-shrink-0">
        <FileText className="w-5 h-5 text-muted-foreground" />
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-medium text-foreground truncate">
          {document.name}
        </h4>
        <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
          <span>{formatFileSize(document.size)}</span>
          <span>•</span>
          <div className="flex items-center gap-1">
            <Layers className="w-3 h-3" />
            <span>{document.chunks} chunks</span>
          </div>
        </div>
        {document.extractionMethod && (
          <span className="inline-block mt-1 text-xs px-2 py-0.5 rounded-full bg-accent text-muted-foreground">
            {document.extractionMethod}
          </span>
        )}
      </div>
      <button
        onClick={() => onDelete(document.id)}
        className="opacity-0 group-hover:opacity-100 p-2 text-muted-foreground hover:text-destructive transition-all"
        aria-label="Delete document"
      >
        <Trash2 className="w-4 h-4" />
      </button>
    </div>
  );
};

export default DocumentCard;

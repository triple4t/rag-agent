import { X, FileText } from 'lucide-react';
import UploadArea from './UploadArea';
import DocumentCard from './DocumentCard';
import type { Document } from '@/types';

interface SidebarProps {
  documents: Document[];
  onUpload: (files: File[]) => Promise<void>;
  onDelete: (id: string) => void;
  isUploading: boolean;
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar = ({
  documents,
  onUpload,
  onDelete,
  isUploading,
  isOpen,
  onClose,
}: SidebarProps) => {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed lg:static inset-y-0 left-0 z-50 w-80 bg-sidebar flex flex-col border-r border-sidebar-border transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-sidebar-border">
          <h2 className="text-lg font-medium text-foreground">Documents</h2>
          <button
            onClick={onClose}
            className="lg:hidden p-2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Upload Area */}
        <div className="p-4">
          <UploadArea onUpload={onUpload} isUploading={isUploading} />
        </div>

        {/* Documents List */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-4">
          {documents.length > 0 ? (
            <div className="space-y-2">
              {documents.map((doc) => (
                <DocumentCard key={doc.id} document={doc} onDelete={onDelete} />
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <FileText className="w-12 h-12 text-muted-foreground/50 mb-3" />
              <p className="text-sm text-muted-foreground">
                No documents uploaded yet
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-sidebar-border">
          <p className="text-xs text-muted-foreground text-center">
            {documents.length} document{documents.length !== 1 ? 's' : ''} loaded
          </p>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;

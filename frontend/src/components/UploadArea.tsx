import { useState, useCallback, DragEvent } from 'react';
import { Upload, Loader2 } from 'lucide-react';

interface UploadAreaProps {
  onUpload: (files: File[]) => Promise<void>;
  isUploading: boolean;
}

const UploadArea = ({ onUpload, isUploading }: UploadAreaProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files).filter(
        (file) => file.type === 'application/pdf'
      );

      if (files.length > 0) {
        await onUpload(files);
      }
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      if (files.length > 0) {
        await onUpload(files);
      }
      e.target.value = '';
    },
    [onUpload]
  );

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer ${
        isDragging
          ? 'border-primary bg-primary/10'
          : 'border-sidebar-border hover:border-muted-foreground'
      }`}
    >
      <input
        type="file"
        accept=".pdf"
        multiple
        onChange={handleFileSelect}
        disabled={isUploading}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
      />
      <div className="flex flex-col items-center gap-2">
        {isUploading ? (
          <Loader2 className="w-8 h-8 text-primary animate-spin" />
        ) : (
          <Upload className="w-8 h-8 text-muted-foreground" />
        )}
        <p className="text-sm font-medium text-foreground">
          {isUploading ? 'Uploading...' : 'Click or drag PDF files here'}
        </p>
        <p className="text-xs text-muted-foreground">Supports PDF documents</p>
      </div>
    </div>
  );
};

export default UploadArea;

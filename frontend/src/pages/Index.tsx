import { useState, useEffect, useCallback } from 'react';
import Sidebar from '@/components/Sidebar';
import ChatInterface from '@/components/ChatInterface';
import { useRAG } from '@/hooks/useRAG';
import { toast } from 'sonner';

const Index = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const {
    documents,
    messages,
    isUploading,
    isLoading,
    fetchDocuments,
    uploadDocuments,
    deleteDocument,
    sendQuery,
  } = useRAG();

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Handle delete with confirmation
  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteDocument(id);
        // Refresh documents list after successful delete
        await fetchDocuments();
      } catch (error) {
        // Error is already handled in useRAG hook
        console.error('Failed to delete document:', error);
      }
    },
    [deleteDocument, fetchDocuments]
  );

  // Handle upload with refresh
  const handleUpload = useCallback(
    async (files: File[]) => {
      try {
        await uploadDocuments(files);
        // Refresh documents list after successful upload
        await fetchDocuments();
      } catch (error) {
        // Error is already handled in useRAG hook
        console.error('Failed to upload documents:', error);
      }
    },
    [uploadDocuments, fetchDocuments]
  );

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        documents={documents}
        onUpload={handleUpload}
        onDelete={handleDelete}
        isUploading={isUploading}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <ChatInterface
        messages={messages}
        onSend={sendQuery}
        isLoading={isLoading}
        hasDocuments={documents.length > 0}
        onMenuClick={() => setSidebarOpen(true)}
      />
    </div>
  );
};

export default Index;
